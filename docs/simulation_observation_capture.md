# Simulation observation capture

The decision observation comes from the native simulator. Capture profiles
control JPEG and H.264 encoding; model-view selection is applied consistently
to decision observations and rollout evaluation.

Mass x Friction uses an independent candidate preview containing the decision
frame and the next two frames under that candidate's action sequence. All three
frames must precede collision. JPEG quality 95 and H.264 CRF 18/GOP 2 reproduce
the dataset's encoding convention; only the decoded first frame is supplied to
the prediction model. No postcollision frame is used to construct the input.

The preview is discarded without advancing the paused execution scene. Each
environment and action uses its own simulator frames, not another environment's
encoded template. Saved prefix observations and contact/action logs identify
exactly which frames contributed to encoding.

Other tasks retain their configured capture policy. An optional history profile
encodes observations available before and at the decision boundary. Execution
video encoding occurs after selection and does not alter the prediction input.
