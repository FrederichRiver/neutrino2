from hmmlearn.hmm import GaussianHMM
from venus.stock_base2 import StockBase
from polaris.mysql8 import GLOBAL_HEADER
import matplotlib.pyplot as plt

event = StockBase(GLOBAL_HEADER)

df = event.select_values('SH600000', 'close_price')
df.columns = ['close_price']
print(df.head(5))

fig = plt.figure()

plt.plot(df['close_price'])
plt.show()
plt.savefig('/home/friederich/Dev/neutrino2/andromeda/andromeda/sh600000.jpg')