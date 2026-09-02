"""GUI smoke test: launch app, load model, open file stream, run briefly."""

import os
import logging
import sys

logging.disable(logging.CRITICAL)

from kivy.clock import Clock

os.environ.setdefault("KIVY_AUDIO", "dummy")
os.environ.setdefault("KIVY_NO_ARGS", "1")


def smoke_test():
    from micro_classifier.gui.app import MicroClassifyApp

    app = MicroClassifyApp()

    processed = {"ticks": 0}

    def on_start():
        assert app.root is not None, "App root not built"
        print("GUI built OK")

        app.load_model(os.path.join("trained_models", "best_microclassifier.pth"))
        print("Model loaded in GUI OK")

        app.start_stream("test_micro.mp4")
        print("Stream started in GUI OK")

        Clock.schedule_interval(
            lambda dt: process(), 1.0 / 15.0
        )
        Clock.schedule_once(finish, 3.0)

    def process():
        app._update_stream(0)
        processed["ticks"] += 1

    def finish(dt):
        app.stop_stream()
        app.stop()
        print(f"GUI ran {processed['ticks']} update ticks")
        print("GUI_SMOKE_OK")

    Clock.schedule_once(lambda dt: on_start(), 0.5)
    app.run()


if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("GUI_SMOKE_FAILED")
        sys.exit(1)