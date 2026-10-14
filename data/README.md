# Data

`sales.parquet` (2.3 MB) is the full **AdventureWorks Sales** sample published by Microsoft in
[microsoft/powerbi-desktop-samples](https://github.com/microsoft/powerbi-desktop-samples/tree/main/AdventureWorks%20Sales%20Sample)
under the MIT licence (notice below), flattened from its star schema by `scripts/build_dataset.py`.

- 121,253 order lines · 31,455 orders · Jul 2017 – Jun 2020 (data stops on 15 Jun 2020)
- 2 channels (Internet, Reseller) · 6 countries · 266 products sold (397 in the catalogue) · 18,484 online customers · 633 resellers

Rebuild it from the original workbook at any time:

```bash
python scripts/build_dataset.py
```

---

MIT License — Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
