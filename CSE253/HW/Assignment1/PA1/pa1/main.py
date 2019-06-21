from utils import *
from dataloader import *
from regression import *
if __name__ == '__main__':
    
    # dataset for logistic regression
    #images, labels, pids, emotion_labels = load_data("happy_maudlin")
    #images, labels, pids, emotion_labels = load_data("anger_surprised")
    
    # dataset for softmax regression
    #images, labels, pids,emotion_labels = load_data("all")
    pid_one_hot = to_onehot(pids)
    
    # remove comments to call functions
    
    #experiment_1 - 2b(i) - Happy vs Maudlin or Anger vs Surprised
    """
    #train_losses, val_losses, acc = logistic_BGD(images, labels, pids, epochs=10, dims_rescaled_data=10, learning_rate=0.01)
    #plot_loss(train_losses.mean(axis=0), val_losses.mean(axis=0), train_losses.std(axis=0),val_losses.std(axis=0))
    """
    
    # experiment_2 - 2b(iii) - Happy vs Maudlin or Anger vs Surprised
    """
    train_losses_1, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.3)
    train_losses_2, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.01)
    train_losses_3, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.001)
    plot_loss_lr(train_losses_1.mean(axis=0),train_losses_2.mean(axis=0),train_losses_3.mean(axis=0),train_losses_1.std(axis=0),train_losses_2.std(axis=0),train_losses_3.std(axis=0))
    """
    
    # experiment_3 - all
    """
    train_losses, val_losses, acc = softmax_BGD(images, labels, pids, epochs=50, dims_rescaled_data=40, learning_rate=0.05)
    plot_loss(train_losses.mean(axis=0), val_losses.mean(axis=0), train_losses.std(axis=0),val_losses.std(axis=0))
    """
    
    # experiment_4 - all
    """
    train_losses_2, val_losses_2, acc_2 = softmax_SGD(images, labels, pids, epochs=50, dims_rescaled_data=40,learning_rate=0.05)
    plot_loss(train_losses_2.mean(axis=0), val_losses_2.mean(axis=0), train_losses_2.std(axis=0),val_losses_2.std(axis=0))
    plot_bgd_sgd_loss(train_losses.mean(axis=0),train_losses_2.mean(axis=0))
    """
    
    # experiment_5 - all
    """
    train_losses_2, val_losses_2, acc_2 = identity_classification(images, pid_one_hot, emotion_labels, epochs=50, dims_rescaled_data=10,learning_rate=0.02)
    plot_loss(train_losses_2.mean(axis=0), val_losses_2.mean(axis=0), train_losses_2.std(axis=0),val_losses_2.std(axis=0))
    """


