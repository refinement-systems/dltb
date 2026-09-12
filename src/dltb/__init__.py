"""dltb: self-iteration feedback-loop drift experiments (image + video).

Shared library:

  dltb.models    model table (ModelSpec/MODELS), loading, fast-fail checks
  dltb.imaging   single-pass execution, frame prep, optical-flow reprojection
  dltb.output    run-directory layout, input sniffing, timelapse assembly
  dltb.args      argparse flag groups shared by the tools

Tools (installed as console scripts; see pyproject.toml):

  dltb-oneshot      single image, single model pass (the anchored fixed point)
  dltb-iterate      free-running image self-iteration + mp4 timelapse
  dltb-continuous   video pipeline simulation (anchored/stateful, tails)
  dltb-klein        the continuous loop restricted to the FLUX.2 klein editors
                    (no --strength; prompt = per-pass edit strength)
"""
