"""نقطه‌ی ورود مشترک اجرای S0/S1 هر خانواده — بدون بارگذاری ماژول خانواده به‌عنوان ``__main__``.

⚠️ **چرا این فایل لازم است.** اگر ``python -m src.models.families.f0X_*`` مستقیم اجرا شود،
آن فایل زیر نام ``"__main__"`` import می‌شود (کدهای سطح ماژول — از جمله حلقه‌ی
``register()`` هر مدل — همان‌جا اجرا می‌شوند). با ``multiprocessing`` روی ``spawn``
(بند 7.7.4 — لازم برای جداسازی ریسه‌ی BLAS، بند ۱.۴ استاندارد اجرا)، هر worker تازه هنگام
bootstrap دوباره همان اسکریپت را زیر نام ``"__main__"`` بارگذاری می‌کند، و اگر
``_init_worker`` صریحاً همان فایل را زیر نام دات‌دار واقعی‌اش هم import کند، **دو import
مستقل از یک فایل در یک process** رخ می‌دهد ⇒ هر ``register()`` دوبار اجرا می‌شود ⇒
``ValueError: model_id تکراری``.

راه‌حل: ماژول خانواده هرگز نباید ``__main__`` باشد. این اسکریپت آن را همیشه با نام
دات‌دار import می‌کند و تابع ``main()``/``run_s1()``اش را صدا می‌زند — برای هر ۱۳ خانواده
یکسان، بدون تکرار argparse در هر ``f0X_*.py``.

اجرا::

    python -m src.models.run_family src.models.families.f01_linear --stage s0
    python -m src.models.run_family src.models.families.f01_linear --stage s1 --jobs 8
"""

import argparse
import importlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="مسیر دات‌دار ماژول خانواده، مثل src.models.families.f01_linear")
    parser.add_argument("--stage", choices=["s0", "s1"], default="s0")
    parser.add_argument("--jobs", type=int, default=None, help="تعداد worker موازی برای S1")
    args = parser.parse_args()

    mod = importlib.import_module(args.module)
    if args.stage == "s0":
        mod.main()
    else:
        mod.run_s1(n_jobs=args.jobs)


if __name__ == "__main__":
    main()
