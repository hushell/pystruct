import numpy as np
import matplotlib.pyplot as plt

def show_prediction(clf,X,Y):
    Y_pred = np.array(clf.predict(X))

    np.set_printoptions(suppress=True)  # suppress scientific notation
    print(clf.w)

    i = 0
    loss = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        y_pred = y_pred.reshape(x.shape[:2])
        loss += np.sum(y != y_pred)
        #if i > 10:
            #continue
        fig, plots = plt.subplots(1, 4)
        plots[0].matshow(y)
        plots[0].set_title("gt")
        plots[1].matshow(np.argmax(x, axis=-1))
        plots[1].set_title("unaries only")
        plots[2].matshow(y_pred)
        plots[2].set_title("prediction")
        loss_augmented = clf.model.loss_augmented_inference(x, y, clf.w)
        loss_augmented = loss_augmented.reshape(y.shape)
        plots[3].matshow(loss_augmented)
        plots[3].set_title("loss augmented")
        for p in plots:
            p.set_xticks(())
            p.set_yticks(())
        plt.show()
        #fig.savefig("data_%03d.png" % i)
        #plt.close(fig)
        i += 1
    print("loss: %f" % loss)
    print("complete loss: %f" % np.sum(Y != Y_pred))

if __name__ == "__main__":
    main()
