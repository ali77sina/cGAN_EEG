import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

g_model = tf.keras.models.load_model('cgan_generator.h5')

npt = 4000
noise = np.random.randn(npt, 5)
labels = np.random.randint(0,2,npt)
preds = g_model.predict((noise,labels))
preds.shape

pos_ind = [num for num,i in enumerate(labels) if i == 1]
neg_ind = [num for num,i in enumerate(labels) if i == 0]
x_pos = preds[pos_ind]
x_neg = preds[neg_ind]
psd_pos = []
psd_neg = []
for i in range(len(x_pos)):
  vals_pos, _ = plt.psd(x_pos[i])
  psd_pos.append(10*np.log(vals_pos))
plt.show()
psd_pos = np.array(psd_pos)

means_pos  = []
std_pos = []

for i in range(129):
  means_pos.append(np.mean(psd_pos[:,i]))
  std_pos.append(np.std(psd_pos[:,i]))
  
fs = np.linspace(0,128,129)
means_pos = np.array(means_pos)
std_pos = np.array(std_pos)
plt.plot(fs, means_pos, color = 'orange')
y1 = means_pos + std_pos
y2 = means_pos - std_pos
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
plt.show()


psd_neg = []
for i in range(len(x_neg)):
  vals_neg, _ = plt.psd(x_neg[i])
  psd_neg.append(10*np.log(vals_neg))
psd_neg = np.array(psd_neg)
plt.show()
means_neg  = []
std_neg = []

for i in range(129):
  means_neg.append(np.mean(psd_neg[:,i]))
  std_neg.append(np.std(psd_neg[:,i]))
  
fs = np.linspace(0,128,129)
means_neg = np.array(means_neg)
std_neg = np.array(std_neg)
plt.plot(fs, means_neg, color = 'orange')
y1 = means_neg + std_neg
y2 = means_neg - std_neg
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
plt.show()

fs = np.linspace(0,128,129)
plt.subplot(221)
plt.plot(fs, means_pos, color = 'orange')
y1 = means_pos + std_pos
y2 = means_pos - std_pos
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
#plt.yscale('log')
plt.title('positive class PSD plot')
plt.subplot(222)
plt.plot(fs, means_neg, color = 'orange')
y1 = means_neg + std_neg
y2 = means_neg - std_neg
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
#plt.yscale('log')
plt.title('negative class PSD plot')
plt.subplot(223)
plt.plot(fs, std_pos)
plt.title('postive class std')
plt.subplot(224)
plt.plot(fs, std_neg)
plt.title('negative class std')
plt.tight_layout()
plt.show()

plt.hist(np.reshape(psd_neg, 261870 ), bins = 500, label = 'negative class', alpha = 0.6, density = True)
plt.hist(np.reshape(psd_pos, 254130), bins = 500, label = 'positive class', alpha = 0.6, density = True)
plt.ylabel('repition')
plt.xlabel('Amp [dB/Hz]')
plt.grid()
plt.legend()
plt.title("Histogram of fake data samples")
plt.show()


pos_ind = [num for num,i in enumerate(y_train) if i == 1]
neg_ind = [num for num,i in enumerate(y_train) if i == 0]
x_pos = x_train[pos_ind]
x_neg = x_train[neg_ind]
psd_pos = []

for i in range(len(x_pos)):
  vals_pos, _ = plt.psd(x_pos[i])
  psd_pos.append(10*np.log(vals_pos))
psd_pos = np.array(psd_pos)

means_pos  = []
std_pos = []

for i in range(129):
  means_pos.append(np.mean(psd_pos[:,i]))
  std_pos.append(np.std(psd_pos[:,i]))

psd_neg = []
for i in range(len(x_neg)):
  vals_neg, _ = plt.psd(x_neg[i])
  psd_neg.append(10*np.log(vals_neg))
psd_neg = np.array(psd_neg)

means_neg  = []
std_neg = []

for i in range(129):
  means_neg.append(np.mean(psd_neg[:,i]))
  std_neg.append(np.std(psd_neg[:,i]))


means_pos = np.array(means_pos)
means_neg = np.array(means_neg)
std_neg = np.array(std_neg)
std_pos = np.array(std_pos)
plt.subplot(121)
plt.plot(fs, means_pos, color = 'orange')
y1 = means_pos + std_pos
y2 = means_pos - std_pos
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
plt.title('Real Positive Class')
plt.subplot(122)
plt.plot(fs, means_neg, color = 'orange')
y1 = means_neg + std_neg
y2 = means_neg - std_neg
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
plt.title('Real Negative Class')
plt.show()


plt.hist(np.reshape(psd_neg, 840177 ), bins = 500, label = 'negative class', alpha = 0.6, density = True)
plt.hist(np.reshape(psd_pos, 280059 ), bins = 500, label = 'positive class', alpha = 0.6, density = True)
plt.ylabel('% data')
plt.xlabel('Amp [dB/Hz]')
plt.grid()
plt.legend()
plt.title("Histogram of real data samples")
plt.show()
