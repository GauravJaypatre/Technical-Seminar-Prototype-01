# Issue: Sample variance uses population N divisor instead of Bessel's correction N-1

**Task ID:** `easy_06`  
**Tier:** `easy`  
**Domain:** `Math / Statistics`  

## Description
`sample_variance(data)` divides by `len(data)` instead of `len(data) - 1` (Bessel's correction), resulting in biased variance estimates. Additionally, it raises ZeroDivisionError when `len(data) < 2` instead of raising a descriptive ValueError.
