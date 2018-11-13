import Brain
from tkinter import *
import random

root = Tk()
MANUAL = True

def main():
    # Initialize the single neuron neural network
    global neural_network
    neural_network = Brain.NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    print('-' * 20)
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value

    #training_inputs = Brain.np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    #training_outputs = Brain.np.array([[0, 1, 1, 0]]).T
    if MANUAL:
        canvas = Canvas(width=800, height=800, bg="white")
        canvas.pack_propagate(0)
        root.config(width=400, height=400)
        root.title("Hot Or Cold NN")
        root.grid()
        for i in range(50000):
            global var
            var = IntVar()
            txtvar = StringVar()
            R = float(random.randint(0, 255))/255
            G = float(random.randint(0, 255))/255
            B = float(random.randint(0, 255))/255
            # translation
            colorval = "#%02x%02x%02x" % (int(R*255), int(G*255), int(B*255))
            text = Label( root, textvariable=txtvar, relief=RAISED, bg=colorval, bd=0, )
            text.config(font=("Courier", 44))
            if neural_network.think(Brain.np.array([R, G, B])) >0.5:
                txtvar.set("HOT!")
            else:
                txtvar.set("COLD!")
            root.configure(background=colorval)
            root.update()
            button_hot = Button(root, text="HOT", command=lambda: teach(1, R, G, B))
            button_hot.pack()
            button_hot.place(x=250, y=150, width=100, height=100)

            button_cold = Button(root, text="COLD", command=lambda: teach(0, R, G, B))
            button_cold.pack()
            button_cold.place(x=50, y=150, width=100, height=100)
            text.pack()
            text.place(x=100, y=0, width=200, height=200)

            root.update()
            button_hot.wait_variable(var)








def teach(output, R, G, B):
    global var
    global neural_network
    var.set(1)
    training_inputs = Brain.np.array([[R, G, B]])

    training_outputs = Brain.np.array([[output]]).T

    # Train the neural network
    neural_network.super_train(training_inputs, training_outputs, 1)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    print('-' * 20)
    print(neural_network.synaptic_weights)

    print("New situation: input data = ", R, G, B)
    print("Output data: ")
    print(neural_network.think(Brain.np.array([R, G, B])))

if __name__ == "__main__":
    main()
    FileEW.file.close()

