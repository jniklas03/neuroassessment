# Report data and figures

Use primary_duration_test.csv for the single primary overall duration comparison.
All other tests are exploratory. See descriptives_overall.csv and
individual_overall_means.csv for corresponding distributions and participant values.
Condition tables use one mean per participant and condition; *_n_trials columns
provide outcome-specific trial counts. participant_flow.csv audits exclusions and
tracking eligibility. trial_audit.csv preserves source participant names.

Total Task Duration is a recording-duration proxy; P1/P2 cannot be derived.
Motion features use wrist-relative fingertip motion and are not physical hand speed.
PNG figures are 300 dpi; PDFs provide vector output. Four-condition lines show
individual subjects; overall boxplots show individual means; duration_changes shows
individual vision/hand changes with median bars. Missing values are not imputed.

Effect directions: MS minus controls; closed minus open; nondominant minus dominant.
Independent effect CIs are percentile bootstrap rank-biserial intervals; paired CIs
are percentile bootstrap median-change intervals. See notebook for definitions.
Participant identifiers are retained in tables; acquisition files are untouched.
