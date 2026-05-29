# Friday, June 5: Training Launch & Log Monitoring

Today we will configure our model hyperparameters in a YAML configuration file, submit the training job to the OSC cluster queue, and write a script to monitor its progress by parsing and plotting the training logs.

---

## Morning Session: Hyperparameters & SLURM Script Setup

### Hyperparameter Configurations

Before launching a deep learning job, we structure our settings in a `.yaml` file. This lets us easily adjust settings without editing the training code:

* `learning_rate`: Controls how large of a step the optimizer takes when updating weights (e.g. `0.001` for the classifier head, `1e-5` for fine-tuning).
* `batch_size`: The number of leaf images shown to the model before updating gradients (e.g., `16` or `32`).
* `epochs`: The number of complete passes through the dataset.
* `weight_decay`: Regularization that prevents the weights from growing too large (which prevents overfitting).

### Launching the Job

When you run `sbatch train_job.sh`, SLURM redirects the terminal output to a log file, normally named `slurm-<job_id>.out`. You can inspect this file in real-time as the model trains:

```bash
# Display the last 20 lines of the log and update dynamically as new lines print
tail -f slurm-1234567.out
```

---

## Afternoon Session: Log Monitoring & Plotting Curves

### Reading Performance from Logs

A typical training loop prints logs like this to stdout:

```text
Epoch 1/5 - Loss: 0.8421 - Accuracy: 0.6540
Epoch 2/5 - Loss: 0.5214 - Accuracy: 0.7820
Epoch 3/5 - Loss: 0.3541 - Accuracy: 0.8870
...
```

Instead of copying these numbers by hand, we write a Python script that reads the log file line-by-line, parses out the numbers using string operations or Regular Expressions (`re` module), and saves them to lists to plot them.

### Plotting Training Curves

A healthy training run shows:

* **Loss Curve**: Decreasing steadily over epochs. If it flatlines immediately, the learning rate might be too high or too low.
* **Accuracy Curve**: Increasing over epochs.

---

## Curated Resources for Day 3

* [Introduction to YAML Syntax](https://yaml.org/spec/1.2.2/)
* [Python Regular Expression (re) Module](https://docs.python.org/3/library/re.html)
* [Matplotlib Customizing Plots](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
