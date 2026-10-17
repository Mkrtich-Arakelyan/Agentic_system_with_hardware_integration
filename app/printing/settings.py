"""Application-owned effective trial-print policy, bound by human approval."""

TRIAL_SIDES = "two-sided-long-edge"
TRIAL_SUMATRA_SETTINGS = "noscale,monochrome,duplexlong,paper=A4,1x"
TRIAL_LP_OPTIONS = ["-n", "1", "-o", "media=A4", "-o", "ColorModel=Gray", "-o", "print-color-mode=monochrome",
                    "-o", "sides=" + TRIAL_SIDES, "-o", "print-scaling=none", "-o", "nofit-to-page"]
TRIAL_PRINT_REMINDER = ("Print settings: A4, grayscale, two-sided (long-edge), Actual size / 100% "
                        "(no fit to page, no scaling).")


def trial_print_settings(args):
    return {**args, "copies": 1, "paper": "A4", "color_mode": "grayscale",
            "sides": TRIAL_SIDES, "scale": "100%"}
