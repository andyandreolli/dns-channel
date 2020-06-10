# dns-channel

A possibly **simple** and **efficient** toolkit to perform DNS, powered with the DNS engine by [Luchini & Quadrio, J. Comp. Phys. (2006)](http://www.sciencedirect.com/science/article/pii/S0021999105002871)

## Changelog

_As a general rule, all of the budget terms calculated by_ uiuj _and_ uiuj_largesmall _have such a sign that, by_ __summing__ _all of them together, one gets 0 (or the residual, more realistically)._ __This is yet to be fixed on other executables (uiuj_ow, uiuj_spectra, possibly read_uiuj).__

Fixes in old _uiuj_ and _uiuj2ascii_ modules:
- tke and its budget are now the trace of the Reynolds stress tensor __already divided by 2__ (as they should be)
- the sign of dissipation has been corrected, so that by summing all the profiles one gets zero (which is, dissipation is now negative for tke and positive for $\langle uv \rangle$ )
- the same treatment in terms of sign has been applied to the mke budget
- every file output by _uiuj2ascii_ now has a header of 1 line which specifies which fields have been used to calculate statistics
- syntax is now:
``` bash
/path/to/exec/uiuj 1 1 localhost start_field end_field step_field
```

Added a module _uiuj_largesmall_ that computes the budget for the large-scale and small-scale fluctuation fields:

- large-small decomposition is defined in `largesmall_def.cpl`; this also refers to a file `largesmall_settings.in` that should be put in the working directory with the fields being postprocessed
- `uiuj_largesmall.cpl` calculates the budget, with the same syntax as _uiuj_
- `uiuj_largesmall2ascii.cpl` converts it to ascii; as for _uiuj2ascii_, a one-line header is included specifying the fields used to calculate average

Added Python scripts to:

- calculate the energy box, with both extended Reynolds decomposition and large/small decomposition
- read the output of _uiuj_ as well as of `dns.in`

