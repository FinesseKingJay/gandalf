

Scorer
=======

`scorer` is a module which builds eval sets, gets output from different systems,
combines systems outputs and evaluates them.

*Extract eval sets*. 
Scorer extract eval set from CLC dev or test part for particular error category
according to CLC-22 categorization. Scorer gathers only sentences which have
at least one error of this particular type. All other annotations are removed
from evalset. So evalset will contain only sentences with at least 1 error (TP).
It works for measuring recall and precision of corrections.
Additionally we measure FP rate on corpus with TN only (coca).

*Gather system outputs*. 
Scorer sends original sentences to different servers using grampy wrappers and
collects outputs of each system.
There are 5 systems so far:
- `Patterns` (without Underpass/One pass)
- `UPC5-high-precision` (Underpass model which was optimized for precision)
- `UPC5-high-recall` (Underpass model which was optimized for recall)
- `OPC-without-filters` (raw One pass model without applying any of filters)
- `OPC-with-filters` (One pass model for Determiner category  with all filters)

*Combine systems output*. 
Apply one among available strategies for combining systems. 
The list of available strategies: 
- `all-agree` - keep suggestion only if all systems produce the same suggestion. 
- `extend-suggestions` - show all suggestions from all systems. 
- `priority-discard` - discard suggestions using custom priority logic. 
- `kenlm` - KenLM choose between conflicting suggestions. 
- `kenlm-orig` - KenLM choose between conflicting suggestions and original sentence. 

*Evaluate systems.* 
Scorer uses m2 scorer for evaluating systems output. 
Scorer uses the following version of m2: Release 3.2 Revision: 22 April 2014 
and it is run locally.

*Results for Determiner category*
Results are available [here](https://docs.google.com/spreadsheets/d/19k78-pycFo31FBLi4eaBHWp7iDc7M5vHAr9KDV7yEeE/edit#gid=0)
Raw data available on kladovka: `/kladovka/projects/ensemble`


