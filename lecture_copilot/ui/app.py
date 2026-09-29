from __future__ import annotations

import time
from queue import Empty, Queue
from tkinter import BooleanVar, messagebox

import customtkinter as ctk

from lecture_copilot.asr.factory import build_asr
from lecture_copilot.audio.capture import LoopbackDevice, default_loopback, list_loopback_devices
from lecture_copilot.config import AppConfig, LectureContext, ensure_dirs
from lecture_copilot.cost import COST
from lecture_copilot.pipeline.events import EventKind, PipelineEvent
from lecture_copilot.pipeline.session import LectureSession
from lecture_copilot.storage import export_ai, export_markdown, init_db, insert_segment
from lecture_copilot.translation.translator import Translator


LANGS = [
    ("auto", "Auto"),
    ("en", "English"),
    ("sk", "Slovak"),
    ("cs", "Czech"),
    ("uk", "Ukrainian"),
    ("ru", "Russian"),
    ("de", "German"),
]

ASR_LOCAL = "Local whisper (CPU)"
ASR_CLOUD = "OpenRouter (cloud)"


class LectureCopilotApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        init_db()
        self.title("Lecture Copilot")
        self.geometry("920x780")
        self.minsize(820, 680)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config = AppConfig()
        self.asr = build_asr(self.config)
        self.translator = Translator(self.config)
        self.events: Queue[PipelineEvent] = Queue()
        self.session: LectureSession | None = None
        self.devices: list[LoopbackDevice] = []
        self._tick_started: float | None = None

        self._build()
        self._load_devices()
        self.after(120, self._poll_events)
        self.after(250, self._tick_clock)

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(16, 8))
        self.rec_label = ctk.CTkLabel(header, text="○ READY", font=ctk.CTkFont(size=18, weight="bold"))
        self.rec_label.pack(side="left")
        self.clock_label = ctk.CTkLabel(header, text="00:00:00", font=ctk.CTkFont(size=18))
        self.clock_label.pack(side="right")
        self.cost_label = ctk.CTkLabel(header, text="витрати: $0.0000", text_color="#7c9")
        self.cost_label.pack(side="right", padx=18)
        self.level_label = ctk.CTkLabel(header, text="звук: ░░░░░░░░░░  тиша", text_color="#aaa")
        self.level_label.pack(side="right", padx=18)

        form = ctk.CTkFrame(self)
        form.pack(fill="x", padx=18, pady=8)
        form.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(form, text="Предмет").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.subject = ctk.CTkEntry(form, placeholder_text="Algorithms")
        self.subject.grid(row=0, column=1, padx=10, pady=8, sticky="ew")

        ctk.CTkLabel(form, text="Тема").grid(row=0, column=2, padx=10, pady=8, sticky="w")
        self.topic = ctk.CTkEntry(form, placeholder_text="Pointers / Dynamic Memory")
        self.topic.grid(row=0, column=3, padx=10, pady=8, sticky="ew")

        ctk.CTkLabel(form, text="Мова лекції").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.lang = ctk.CTkComboBox(form, values=[f"{c} — {n}" for c, n in LANGS], width=180)
        self.lang.set("auto — Auto")
        self.lang.grid(row=1, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(form, text="Whisper").grid(row=1, column=2, padx=10, pady=8, sticky="w")
        self.model = ctk.CTkComboBox(form, values=["tiny", "base", "small", "medium"], width=140)
        self.model.set("small")
        self.model.grid(row=1, column=3, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(form, text="Джерело звуку").grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.device_box = ctk.CTkComboBox(form, values=["(шукаю пристрої...)"], width=420)
        self.device_box.grid(row=2, column=1, columnspan=3, padx=10, pady=8, sticky="ew")

        ctk.CTkLabel(form, text="Терміни (через кому)").grid(row=3, column=0, padx=10, pady=8, sticky="w")
        self.keywords = ctk.CTkEntry(
            form,
            placeholder_text="pointer, malloc, calloc, realloc, free, stack, heap",
        )
        self.keywords.grid(row=3, column=1, columnspan=3, padx=10, pady=8, sticky="ew")

        ctk.CTkLabel(form, text="ASR двигун").grid(row=4, column=0, padx=10, pady=8, sticky="w")
        self.asr_engine = ctk.CTkComboBox(form, values=[ASR_LOCAL, ASR_CLOUD], width=220)
        self.asr_engine.set(ASR_CLOUD if self.config.asr_provider == "openrouter" else ASR_LOCAL)
        self.asr_engine.grid(row=4, column=1, columnspan=3, padx=10, pady=8, sticky="w")

        self.slides_var = BooleanVar(value=self.config.slide_enabled)
        self.slides_toggle = ctk.CTkCheckBox(
            form, text="Захоплення слайдів / дошки", variable=self.slides_var
        )
        self.slides_toggle.grid(row=5, column=0, columnspan=2, padx=10, pady=8, sticky="w")
        self.slide_provider_box = ctk.CTkComboBox(
            form, values=["auto", "openrouter", "tesseract", "none"], width=180
        )
        self.slide_provider_box.set(self.config.slide_provider)
        self.slide_provider_box.grid(row=5, column=2, columnspan=2, padx=10, pady=8, sticky="w")

        self.status = ctk.CTkLabel(form, text=self._mode_text(), text_color="#9ad")
        self.status.grid(row=6, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="w")

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=18, pady=4)
        self.start_btn = ctk.CTkButton(btns, text="START LECTURE", command=self.toggle_record, width=160)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.pause_btn = ctk.CTkButton(btns, text="Pause", command=self.toggle_pause, width=110, state="disabled")
        self.pause_btn.pack(side="left", padx=8)
        self.mark_btn = ctk.CTkButton(btns, text="Mark important", command=self.mark_important, width=140, state="disabled")
        self.mark_btn.pack(side="left", padx=8)
        self.save_btn = ctk.CTkButton(btns, text="Save note", command=self.save_note, width=120, state="disabled")
        self.save_btn.pack(side="left", padx=8)

        orig_frame = ctk.CTkFrame(self)
        orig_frame.pack(fill="both", expand=True, padx=18, pady=(10, 6))
        ctk.CTkLabel(orig_frame, text="ORIGINAL", anchor="w").pack(fill="x", padx=12, pady=(8, 0))
        self.original = ctk.CTkTextbox(orig_frame, height=130, wrap="word")
        self.original.pack(fill="both", expand=True, padx=12, pady=8)

        uk_frame = ctk.CTkFrame(self)
        uk_frame.pack(fill="both", expand=True, padx=18, pady=(0, 6))
        ctk.CTkLabel(uk_frame, text="🇺🇦 УКРАЇНСЬКОЮ", anchor="w").pack(fill="x", padx=12, pady=(8, 0))
        self.ukrainian = ctk.CTkTextbox(uk_frame, height=130, wrap="word")
        self.ukrainian.pack(fill="both", expand=True, padx=12, pady=8)

        slides_frame = ctk.CTkFrame(self)
        slides_frame.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        ctk.CTkLabel(slides_frame, text="📊 СЛАЙДИ / ДОШКА", anchor="w").pack(fill="x", padx=12, pady=(8, 0))
        self.slides = ctk.CTkTextbox(slides_frame, height=110, wrap="word")
        self.slides.pack(fill="both", expand=True, padx=12, pady=8)

    def _mode_text(self) -> str:
        engine = self.asr_engine.get() if hasattr(self, "asr_engine") else ASR_LOCAL
        if self.config.translation_enabled:
            return f"Режим: {engine} + переклад ({self.translator.provider_label()})"
        return "Режим: без перекладу — додай GEMINI_API_KEY або OPENROUTER_API_KEY у .env"

    def _load_devices(self) -> None:
        try:
            self.devices = list_loopback_devices()
            names = [d.name for d in self.devices] or ["(немає loopback)"]
            self.device_box.configure(values=names)
            try:
                default = default_loopback()
                self.device_box.set(default.name)
            except Exception:
                self.device_box.set(names[0])
        except Exception as exc:
            self.device_box.configure(values=[str(exc)])
            self.device_box.set(str(exc))

    def _selected_device(self) -> LoopbackDevice | None:
        name = self.device_box.get()
        for d in self.devices:
            if d.name == name:
                return d
        return None

    def _context(self) -> LectureContext:
        lang = self.lang.get().split("—", 1)[0].strip()
        kws = [p.strip() for p in self.keywords.get().split(",") if p.strip()]
        return LectureContext(
            subject=self.subject.get().strip() or "Lecture",
            topic=self.topic.get().strip(),
            source_language=lang,
            keywords=kws,
        )

    def toggle_record(self) -> None:
        if self.session is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        self.config.whisper_model = self.model.get()
        COST.reset()
        self.cost_label.configure(text="витрати: $0.0000")
        self.config.asr_provider = (
            "openrouter" if self.asr_engine.get() == ASR_CLOUD else "whisper"
        )
        self.asr = build_asr(self.config)
        self.config.slide_enabled = bool(self.slides_var.get())
        self.config.slide_provider = self.slide_provider_box.get()
        self.original.delete("1.0", "end")
        self.ukrainian.delete("1.0", "end")
        self.slides.delete("1.0", "end")
        ctx = self._context()
        self.session = LectureSession(
            config=self.config,
            ctx=ctx,
            asr=self.asr,
            translator=self.translator,
            events=self.events,
            device=self._selected_device(),
        )
        self.session.start()
        self._tick_started = time.monotonic()
        self.rec_label.configure(text="● RECORDING", text_color="#e55")
        self.start_btn.configure(text="STOP")
        self.pause_btn.configure(state="normal", text="Pause")
        self.mark_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.status.configure(
            text=f"{self.asr.label}: перший старт локальної моделі може тривати 1–5 хв. Чекай «готовий», не STOP."
        )

    def _stop(self) -> None:
        if self.session is None:
            return
        session = self.session
        self.session = None
        session.stop()
        self._tick_started = None
        self.rec_label.configure(text="○ READY", text_color="white")
        self.start_btn.configure(text="START LECTURE")
        self.pause_btn.configure(state="disabled", text="Pause")
        self.status.configure(text="Лекцію зупинено. AI-бандл збережено.")
        try:
            path = export_markdown(session.lecture_id)  # type: ignore[arg-type]
            res = export_ai(session.lecture_id)  # type: ignore[arg-type]
            self.status.configure(text=f"Готово: {path.parent} (lecture.md + AI-бандл)")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def toggle_pause(self) -> None:
        if self.session is None:
            return
        paused = self.pause_btn.cget("text") != "Resume"
        self.session.set_paused(paused)
        self.pause_btn.configure(text="Resume" if paused else "Pause")
        self.rec_label.configure(text="⏸ PAUSED" if paused else "● RECORDING")

    def mark_important(self) -> None:
        if self.session is None or self.session.lecture_id is None:
            return
        elapsed = time.monotonic() - (self._tick_started or time.monotonic())
        insert_segment(
            self.session.lecture_id,
            elapsed,
            elapsed,
            "[IMPORTANT]",
            "[ВАЖЛИВО]",
        )
        self.original.insert("end", "\n★ [IMPORTANT]\n")
        self.ukrainian.insert("end", "\n★ [ВАЖЛИВО]\n")

    def save_note(self) -> None:
        lecture_id = self.session.lecture_id if self.session else None
        if lecture_id is None:
            messagebox.showinfo("Save note", "Спочатку запусти лекцію.")
            return
        try:
            path = export_markdown(lecture_id)
            res = export_ai(lecture_id)
            messagebox.showinfo(
                "Save note",
                "Готово. Для ШІ/агента:\n"
                f"{res['pack']}\n{res['json']}\n{res['manifest']}\n\n"
                f"Для людини:\n{path}",
            )
        except Exception as exc:
            messagebox.showerror("Save note", str(exc))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            self._handle(event)
        self.after(120, self._poll_events)

    def _handle(self, event: PipelineEvent) -> None:
        if event.kind == EventKind.STATUS:
            self.status.configure(text=event.message, text_color="#9ad")
        elif event.kind == EventKind.ERROR:
            self.status.configure(text=event.message, text_color="#f88")
        elif event.kind == EventKind.LEVEL:
            filled = min(10, int(event.level * 80))
            bar = "█" * filled + "░" * (10 - filled)
            hint = event.message or ("тиша — інший пристрій або звук не грає")
            self.level_label.configure(text=f"звук: {bar}  {hint}")
        elif event.kind == EventKind.SEGMENT:
            if event.translated:
                self.ukrainian.insert("end", event.translated + "\n")
                self.ukrainian.see("end")
            else:
                self.original.insert("end", event.original + "\n")
                self.original.see("end")
        elif event.kind == EventKind.SLIDE:
            stamp = time.strftime("%H:%M:%S", time.gmtime(event.start_time))
            block = f"\n📊 {stamp} — {event.message}\n"
            if event.translated:
                block += f"{event.translated}\n"
            elif event.original:
                block += f"{event.original[:300]}\n"
            self.slides.insert("end", block)
            self.slides.see("end")

    def _tick_clock(self) -> None:
        if self._tick_started is not None and (self.session is None or not self.session.is_paused()):
            elapsed = int(time.monotonic() - self._tick_started)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            self.clock_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.cost_label.configure(text=f"витрати: {COST.summary()}")
        self.after(250, self._tick_clock)

    def on_close(self) -> None:
        if self.session is not None:
            self.session.stop()
        self.translator.close()
        close = getattr(self.asr, "close", None)
        if callable(close):
            close()
        self.destroy()


def run() -> None:
    app = LectureCopilotApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
