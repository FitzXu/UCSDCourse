1. For data load, remove comments of 
	
	images, labels, pids, emotion_labels = load_data("happy_maudlin") or
	images, labels, pids, emotion_labels = load_data("anger_surprised") or
	images, labels, pids, emotion_labels = load_data("all")

   to load Happy vs Maudlin or Anger vs Surprised or all dataset.

2. For Logistic Regression part, load Happy vs Maudlin or Anger vs Surprised first, then do following things:
	
   2b(i) and 2c(i), remove comments of:

	train_losses, val_losses, acc = logistic_BGD(images, labels, pids, epochs=10, dims_rescaled_data=10, learning_rate=0.01)
    	plot_loss(train_losses.mean(axis=0), val_losses.mean(axis=0), train_losses.std(axis=0),val_losses.std(axis=0))

   2b(iii), remove comments of:

	train_losses_1, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.3)
    	train_losses_2, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.01)
    	train_losses_3, _, _ = logistic_SGD(images, labels, pids, epochs=10, dims_rescaled_data=10,learning_rate=0.001)
	plot_loss_lr(train_losses_1.mean(axis=0),train_losses_2.mean(axis=0),train_losses_3.mean(axis=0),train_losses_1.std(axis=0),train_losses_2.std(axis=0),train_losses_3.std(axis=0))

   2b(ii), 2c(ii) and 2c(iii):

	results in report
   
3. For Softmax Regression, load all dataset first, then do following things:

   3(a) Evaluate your network on all six emotions, remove comments of:

	train_losses, val_losses, acc = softmax_BGD(images, labels, pids, epochs=50, dims_rescaled_data=40, learning_rate=0.05)
    	plot_loss(train_losses.mean(axis=0), val_losses.mean(axis=0), train_losses.std(axis=0),val_losses.std(axis=0))

   3(b) Batch versus stochastic gradient descent and 3(c) Visualize the weights, remove comments of:

	train_losses_2, val_losses_2, acc_2 = softmax_SGD(images, labels, pids, epochs=50, dims_rescaled_data=40,learning_rate=0.05)
    	plot_loss(train_losses_2.mean(axis=0), val_losses_2.mean(axis=0), train_losses_2.std(axis=0),val_losses_2.std(axis=0))
    	plot_bgd_sgd_loss(train_losses.mean(axis=0),train_losses_2.mean(axis=0))

   3(d) Extra Credit, remove comments of:

	train_losses_2, val_losses_2, acc_2 = identity_classification(images, pid_one_hot, emotion_labels, epochs=50, dims_rescaled_data=10,learning_rate=0.02)
    	plot_loss(train_losses_2.mean(axis=0), val_losses_2.mean(axis=0), train_losses_2.std(axis=0),val_losses_2.std(axis=0))



