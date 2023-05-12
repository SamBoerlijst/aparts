Frequently wondered inqueries
+++++++++++++++++++++++++++++++++++++++++++++++++

**When I use the tagging functions multiple times, will articles which already have been tagged be skipped?**

- Yes, to improve efficiency keylist_search automatically checks whether a file is present in the output csv file and skippes file shat have been indexed.

**How may I improve results?**

- A manual check to remove artifacts from the generated keywords is advised. Artifacts may be produced in case of typesetting issues (mainly in older pdf files), decoding issues or words containing special characters like chemical compounds.

**I get decoding errors for certain documents during keylist_search().**

- PDF encoding is not standardized and although this package automatically uses common decoding fixes, some articles might still return errors. In this case you may need to manually get rid of unicode characters in the respective txt file with an ascii converter like https://onlineunicodetools.com/convert-unicode-to-ascii.

**I got unknown widths|multiple definitions|Unexpected escaped string errors during pdf2txt conversion. What happened?**

- This happens when the contents of a pdf cannot properly be read. This may be caused by files being corrupt or consisting of scanned pages as may be the case for older pdf files.

**How long will the process take?**

- Several functions may be time consuming. The scholarly lookups for article or author might take several minutes, keyword generation about 10 minutes for 100 records and keylist_search may take about 10 minutes for 500 files.
