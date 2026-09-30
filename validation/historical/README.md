# Historical validation archive

These files are prior-series results, copied unchanged from the planning checkout
`28b6678`. The manifest fingerprints every snapshot. That checkout commit is not
the generation commit of every measurement; use the provenance inside a file when
available, and treat missing generation provenance as unknown. Old reports and
commands are retained for history and are not claims about the revised DYN series.
Current commands and results are one directory above. Do not run the old commands
against the revised benchmark.

The profile-before-cache-selection-fix-* files preserve earlier audit-only
profiles whose initializer wrapper caused the core builder to select uncached
inputs. Their mathematical outputs agree with corrected profiles; use current
results/ for owned-cache transport measurements. Ordinary timing rows were
unaffected because they do not install profile hooks.
