import os

from dotenv import load_dotenv
from edgar import set_identity, Company

load_dotenv()
set_identity(f"Srikar {os.environ['EDGAR_CONTACT_EMAIL']}")


msft = Company("AAPL")                          # resolves ticker -> CIK for you
filings = msft.get_filings(form="10-K")         # all 10-Ks
latest_10k = filings.latest()                   # most recent one
print(latest_10k)
tenk = latest_10k.obj()        # a structured 10-K object
text = latest_10k.text()       # full plain text
print(len(text))               # should be large — hundreds of thousands of chars
# print(text[:1000])    
print(tenk.sections)           # eyeball the first chunk

risk = tenk.risk_factors         # or tenk["Item 1A"]
print(len(risk))
print(risk[:1500])      

fin = tenk.financials
print(fin.income_statement())    # revenue, net income, etc. as structured data         # notice: clean, scoped to just risk factors
# print(dir(latest_10k))