import ast
import matplotlib.pyplot as plt

history = None
with open('historyDict.txt', 'r') as f:
	history = f.read()

history = ast.literal_eval(history)

plt.figure()
plt.plot(range(1, 41), history['loss'], label='Train loss')
plt.plot(range(1, 41), history['val_loss'], label='Test loss')
plt.legend()
plt.title("Loss across epochs")
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(range(1, 41), history['accuracy'], label='Train accuracy')
plt.plot(range(1, 41), history['val_accuracy'], label='Test accuracy')
plt.legend()
plt.title("Accuracy across epochs")
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
