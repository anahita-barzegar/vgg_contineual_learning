import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot


def plot_images(images, labels, im_type, imagination_level):
    labels_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                   8: 'ship', 9: 'truck'}
    for i, img in enumerate(images[:20]):
        img = img.cpu()
        # Convert the tensor to a NumPy array
        numpy_image = img.numpy()

        # Transpose the array to [height, width, channels] for plotting
        if imagination_level == 'data':
            numpy_image = np.transpose(numpy_image, (1, 2, 0))
        else:
            numpy_image = numpy_image.transpose(1, 2, 0)

        plt.imshow(numpy_image)

        # Plot the image
        plt.axis('off')  # Turn off axis labels
        plt.savefig(f'data/imaginations/{imagination_level}_{im_type}_{labels_dict[labels[i].item()]}.png')


def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims == 0 else n_params * dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40 * "-" + title + 40 * "-")
    print(model)
    print(90 * "-")
    _ = count_parameters(model)
    print(90 * "-")


def plot_results(data, first_axis, second_axis, subject):
    # Extract epochs and MSE values from the list of dictionaries
    f1 = [d[first_axis] for d in data]
    f2 = [d[second_axis] for d in data]
    assert len(f1) == len(f2), "f1 and f2 lengths do not match!"

    plt.figure()

    # Plot the MSE values over epochs
    plt.plot(f1, f2, marker='o', linestyle='-')
    plt.title(f'{first_axis.upper()} vs {second_axis.upper()}')
    plt.xlabel(f'{first_axis.upper()}')
    plt.ylabel(f'{second_axis.upper()}')
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(f'/home/anahita/personal_projects/cl/my_contineual_learning/data/results/{subject}_{first_axis.lower()}_vs_{second_axis.lower()}_plot.png')


def plot_model(x, model, plot_name):
    make_dot(model(x), params=dict(model.named_parameters())).render("model_architecture_{plot_name}", format="png",
                                                                     cleanup=True)
