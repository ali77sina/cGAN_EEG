import tensorflow as tf
import numpy as np
import sklearn as sk

x_train = np.load('x_train_256.npy')
y_train = np.load('y_train_256.npy')
x_train, y_train = sk.utils.shuffle(x_train, y_train)

def make_gen_model(latent_dim = 5, n_outputs=256):
  latent_vec = tf.keras.layers.Input(latent_dim, name = 'latent vector')
  cond = tf.keras.layers.Input(1, name = 'conditional input')
  out_cond = tf.keras.layers.Dense(4, activation='relu', kernel_initializer='he_uniform', name = 'hidden_conditional_layer')(cond)
  new_input = tf.keras.layers.concatenate([out_cond, latent_vec], name = 'mixed_input')
  x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', name = 'hidden1_g', input_shape = (latent_dim+4,))(new_input)
  x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', name = 'hidden2_d')(x)
  x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', name = 'hidden3_d')(x)
  output = tf.keras.layers.Dense(n_outputs, activation='linear', name = 'generated_output')(x)
  model = tf.keras.Model(inputs = (latent_vec, cond), outputs = output, name = 'g_model')
  return model
  
  
def make_disc_model(n_inputs=256):
  input_data = tf.keras.layers.Input(n_inputs, name = 'input vector')
  cond = tf.keras.layers.Input(1, name = 'conditional input')
  out_cond = tf.keras.layers.Dense(4, activation='relu', kernel_initializer='he_uniform', name = 'hidden_conditional_layer')(cond)
  new_input = tf.keras.layers.concatenate([out_cond, input_data], name = 'mixed_input')
  x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', name = 'hidden1_d')(new_input)
  x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', name = 'hidden2_d')(x)
  x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', name = 'hidden3_d')(x)
  output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform', name = 'output')(x)
  model = tf.keras.Model(inputs = (input_data, cond), outputs = output, name = 'model_d')
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
  
  
def make_gan_model(generator, discriminator):
  discriminator.trainable = False
  print(1)
  gen_label, gen_noise = generator.input
  print(2)
  gen_output = generator.output
  print(3)
  print(gen_output.shape)
  print(gen_label.shape)
  print(gen_noise.shape)
  gan_output = discriminator([gen_output, gen_noise])
  print(4)
  model = tf.keras.Model([gen_label, gen_noise], gan_output, name = 'gan_mod')
  print(5)
  opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model
  
  
def generate_real_samples(n_samples):
  inds = np.random.randint(0,len(x_train),n_samples)
  # split into images and labels
  X, labels = x_train[inds], y_train[inds]
  # generate class labels
  y = np.ones((n_samples, 1))
  return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(n_samples, latent_dim = 5, n_classes=2):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = np.random.randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
  # generate points in latent space
  # z_input, labels_input = generate_latent_points(latent_dim, n_samples)
  # print(z_input.shape)
  # print(labels_input.shape)
  noise = np.random.randn(n_samples, 5)
  labels = np.random.randint(0,2,n_samples)
  # predict outputs
  images = generator.predict((noise, labels))
  # create class labels
  y = np.zeros((n_samples, 1))
  return [images, labels], y
  
  
def train(g_model, d_model, gan_model, latent_dim, n_epochs=100, n_batch=128):
  bat_per_epo = 8
  half_batch = int(n_batch / 2)
  for i in range(n_epochs):
    for j in range(bat_per_epo):
      [X_real, labels_real], y_real = generate_real_samples(half_batch)
      d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
      [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
      z_input = np.random.randn(n_batch, 5)
      labels_input = np.random.randint(0,2,n_batch)
      y_gan = np.ones(n_batch)
      g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
      print('>%d, %d/%d, real=%f, fake=%f gan=%f' %
        (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
  g_model.save('cgan_generator.h5')
  

latent_dim = 5
d_model = make_disc_model()
g_model = make_gen_model(latent_dim)
gan_model = make_gan_model(g_model, d_model)
train(g_model, d_model, gan_model, latent_dim, n_epochs = 5000)
