from dataloader import *
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def show_six_emotions():
    """
    select 6 different emotions [h, m, s, f, a, d]
    from the CAFE directory
    :return: None
    """
    images, labels = load_data_original()
    emot_list = ["h", "m", "s", "f", "a", "d"]
    show_list = []
    for emot in emot_list:
        for index, name in enumerate(labels):
            if emot in name and "ht" not in name:
                show_list.append((images[index],name))
                break
    fig = plt.figure()
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)
    for index, pair in enumerate(show_list):
        plt.subplot(2,3,index+1)
        plt.imshow(pair[0], cmap ='gray')
        plt.title(pair[1])
        plt.axis("off")
    #plt.show()
    plt.savefig("./images/six_face.png")


def create_train_set():
    """
    creating a training set including the image
    name and corresponding label
    :return: list of pairs of image name and the label. e.g. (array,"a")
    """
    images, labels = load_data()
    print(images[0].shape[0])
    newImages = np.zeros((1,(images[0].shape[0]*images[0].shape[1])))
    newLabels = np.zeros((1,1))

    # filter out emotion happy with teeth and neural faces
    for index, name in enumerate(labels):
        if "_ht" not in name and "_n" not in name:
            newImages = np.vstack((newImages,images[index].flatten()))
            newLabels = np.vstack((newLabels, labels[index].split("_")[1][0]))

    newImages = newImages[1:, :]
    newLabels = newLabels[1:, :]
    print("================================")
    print("Created a trainSet with size: {}".format(newImages.shape[0]))
    print("================================")
    return newImages, newLabels


def pca(data, dims_rescaled_data):
    """
    :param data:
    :param dims_rescaled_data:
    :return: eigen_vectors ,eigen_vals, transformed_data
    """
    # mean center the data
    data /= 255
    N = data.shape[0]
    train_mean = np.mean(data, axis=0)
    data -= train_mean
    #data /= np.std(data)

    # compute the covariance matrix
    covariance = (1/data.shape[0])*np.dot(data, data.T)

    # compute eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eigh(covariance)
    # sort eigenvalue in decreasing order
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # new_evecs 91200 * 60
    new_evecs = np.dot(data.T, evecs)
    #standardize the eigenvecors to make each of their norm is unityw
    for i in range(evals.shape[0]):
        new_evecs[:,i] = new_evecs[:,i]/np.sqrt(np.abs(N*evals[i]))

    # new_traindata 60 x dims_rescaled_data = 60*91200 x 91200*10
    new_data = np.dot(data, new_evecs[:, :dims_rescaled_data])

    #new_data -= np.mean(new_data, axis=0)
    #new_data /= np.std(new_data)
    #show_eigenface(new_evecs)
    return new_evecs[:, :dims_rescaled_data], new_data, train_mean

def pca_fit(eigen_vec, data, train_mean):
    """
    :param eigen_vec: original_dimension x projected_dimension
    :param eigen_val: projected dimension
    :param data: number x original dimension
    :return: projected data
    """
    data -= train_mean
    data /= np.std(data)
    new_data = np.dot(data, eigen_vec)
    new_data -= np.mean(new_data, axis=0)
    new_data /= np.std(new_data)
    return new_data

def show_eigenface(data):
    """
    :param eigen_vecs:
    :return: None,just show eigenfaces
    """
    data -= np.mean(data, axis=0)
    data /= np.std(data)

    # compute the covariance matrix
    covariance = np.dot(data, data.T)

    # compute eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eigh(covariance)
    # sort eigenvalue in decreasing order
    evals = np.sqrt(np.abs(evals[::-1]))
    evecs = evecs[:, ::-1]

    # new_evecs 91200 * 60
    new_evecs = np.dot(data.T, evecs)
    for i in range(evals.shape[0]):
        new_evecs[:,i] = new_evecs[:,i]/evals[i]

    eigen_faces = new_evecs[:, :6]
    fig = plt.figure()
    fig.tight_layout()
    for index in range(6):
        plt.subplot(2, 3, index + 1)
        plt.imshow(eigen_faces[:, index].reshape((380, 240)), cmap='gray')
        plt.title("Eigenfaces {}    ".format(index + 1))
        plt.axis('off')
    plt.savefig("./images/eigenfaces.png")
    plt.show()

def split_data(X, Y, pid, t, dims_rescaled_data):
    """
    split dataset
    """
    # split test set from X
    test_idx = list(np.where(pid == t)[0])
    test_X = X[test_idx]
    test_Y = Y[test_idx]

    # random pick val set from X, test_X exclusive
    idx = list(np.random.permutation(10))
    idx.remove(t)
    val_idx = list(np.where(pid == idx[0])[0])
    val_X = X[val_idx]
    val_Y = Y[val_idx]

    idx.remove(idx[0])
    train_idx = []
    for i in range(len(idx)):
        train_idx.extend(list(np.where(pid == idx[i])[0]))
        train_X = X[train_idx]
        train_Y = Y[train_idx]

    evec, train_pca, train_mean = pca(train_X, dims_rescaled_data)
    # print("std",np.std(train_pca))
    val_pca = pca_fit(evec, val_X, train_mean)
    test_pca = pca_fit(evec, test_X, train_mean)

    return train_pca, train_Y, val_pca, val_Y, test_pca, test_Y, evec

def calculate_accuracy_testset(X, Y, W, mode):
    """
    :param X: data on testset
    :param Y: label on testset
    :param W: weights the best
    :return: accuracy (probability)
    """

    if mode == "binary":
        acc = 0.0
        for i in range(0, len(Y)):
            prob = sigmoid(np.dot(X[i], W))
            if (prob >= 0.5 and Y[i] == 1) or (prob < 0.5 and Y[i] == 0):
                acc += 1
    elif mode == "multiple":
        acc = 0.0
        for i in range(0, len(Y)):
            classid = np.argmax(Y[i])
            an = np.exp(np.dot(X[i], W))
            yn = (an)/np.expand_dims(np.sum(an), axis=1)
            class_pred = np.argmax(yn)
            if classid == class_pred:
                acc += 1
    return acc / Y.shape[0]


def plot_loss(train_losses, val_losses, train_std, val_std):
    
    plt.figure()
    x = [t + 1 for t in range(train_losses.shape[0])]
    plt.plot(x, train_losses, label='train loss', color='r')
    plt.plot(x, val_losses, label='holdout loss', color='y')
    if len(x) == 50:
        index = [9, 19, 29, 39, 49]
    else:
        index = [1, 3, 7, 9]
    plt.errorbar([t+1 for t in index], train_losses[index], yerr=train_std[index], fmt="o",ecolor='r', mfc='red')
    plt.errorbar([t+1 for t in index], val_losses[index], yerr=val_losses[index], fmt="o",ecolor='y', mfc='yellow')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid(True, linestyle = "--", color = "gray", linewidth = 0.5)
    plt.legend()
    #plt.savefig("Report/images/as_loss_recog.png")
    plt.show()
    
def plot_loss_lr(train_loss_1, train_loss_2, train_loss_3, std_1, std_2, std_3):
    
    plt.figure()
    x = [t + 1 for t in range(train_loss_1.shape[0])]
    plt.plot(x, train_loss_1, label='learning rate = 0.001', color='r')
    plt.plot(x, train_loss_2, label='learning rate = 0.01', color='y')
    plt.plot(x, train_loss_3, label='learning rate = 0.3', color='b')
    if len(x) == 50:
        index = [9, 19, 29, 39, 49]
    else:
        index = [1, 3, 7, 9]
    plt.errorbar([t+1 for t in index], train_loss_1[index], yerr=std_1[index], fmt="o",ecolor='r', mfc='red')
    plt.errorbar([t+1 for t in index], train_loss_2[index], yerr=std_2[index], fmt="o",ecolor='y', mfc='yellow')
    plt.errorbar([t+1 for t in index], train_loss_3[index], yerr=std_3[index], fmt="o",ecolor='b', mfc='blue')
    plt.xlabel('Epochs')
    plt.ylabel("Training Loss")
    plt.grid(True, linestyle = "--", color = "gray", linewidth = 0.5)
    plt.legend()
    #plt.savefig("Report/images/hm_loss_recog_lr.png")
    plt.show()
    
def confusion(X, Y, W):
    matrix = np.zeros((Y.shape[1],Y.shape[1]))
    for i in range(0, len(Y)):
        classid = np.argmax(Y[i])
        an = np.exp(np.dot(X[i], W))
        yn = (an) / np.expand_dims(np.sum(an), axis=1)
        class_pred = np.argmax(yn)
        matrix[classid,class_pred] +=1
    return matrix


def plot_confusion_matrix(cm):
    labels = ['a', 'd', 'f', 'h', 'm', 's']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
    #plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    #plt.savefig("./images/confusion_bgd.png")


def visual_weights(weight, eigenvec):
    visual = np.dot(weight.T, eigenvec.T)
    labels = list(range(weight.shape[1]))
    fig = plt.figure()
    fig.tight_layout()
    for i in range(len(labels)):
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(visual[i,:].reshape((380,240)),cmap='gray')
        plt.title("Face: {}".format(labels[i]))
    #plt.savefig("./images/visualization2.png")
    plt.show()

def plot_bgd_sgd_loss(train_losses, train_losses2):
    plt.figure()
    x = [t + 1 for t in range(train_losses.shape[0])]
    plt.plot(x, train_losses, label='train loss BGD', color='r')
    plt.plot(x, train_losses2, label='train loss SGD', color='b')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #plt.savefig("./images/bgd_sgd.png")


def to_onehot(label):
    number_class = np.max(label)+1
    onehot_label = np.zeros((label.shape[0], number_class))
    for i in range(label.shape[0]):
        class_id = label[i]
        onehot_label[i,class_id] = 1
    return onehot_label


def split_for_identity(X, Y, pid, t, dims_rescaled_data):
    test_idx = list(np.where(pid == t)[0])
    test_X = X[test_idx]
    test_Y = Y[test_idx]

    # random pick val set from X, test_X exclusive
    idx = list(np.random.permutation(6))
    idx.remove(t)
    val_idx = list(np.where(pid == idx[0])[0])
    val_X = X[val_idx]
    val_Y = Y[val_idx]

    idx.remove(idx[0])
    train_idx = []
    for i in range(len(idx)):
        train_idx.extend(list(np.where(pid == idx[i])[0]))
        train_X = X[train_idx]
        train_Y = Y[train_idx]

    evec, train_pca, train_mean = pca(train_X, dims_rescaled_data)
    val_pca = pca_fit(evec, val_X, train_mean)
    test_pca = pca_fit(evec, test_X, train_mean)

    return train_pca, train_Y, val_pca, val_Y, test_pca, test_Y, evec