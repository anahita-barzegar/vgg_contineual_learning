import torch
import torch.nn as nn
import torch.optim as optim

from vqvae import VQVAE
from vgg_replay import ShuffleDataset
from torch.utils.data import Dataset, DataLoader

from scipy.stats import multivariate_normal
import general_utils
import csv
import copy

import os
import gc
import shutil

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"


def train_classifier(model, traindata, testdata, num_classes, channel, imagination_level, imagination_param,
                     task_number, epoch_number, model_name, dataset,
                     sleep_param, replay_iter_param, contineual_learning, replay_batch_number, replay_batch_size):
    # general_utils.plot_model(traindata[0], model, 'classifier')
    general_utils.print_model_info(model)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Example usage
    print("Initial model:", model)
    if sleep_param or contineual_learning:
        if imagination_level == 'data':
            x_set, y_set = traindata.dataset.data, traindata.dataset.targets
            shuffle_train_dataset = ShuffleDataset(x_set, y_set, is_numpy=True)
            shuffle_test_dataset = ShuffleDataset(testdata.dataset.data, testdata.dataset.targets, is_numpy=True)
            traindata = shuffle_train_dataset
            testdata = shuffle_test_dataset
        elif imagination_level == 'enc_b':
            # Step 1: Extract data and labels
            x_set = [item[0] for item in traindata]  # Extract all first items (64x data)
            y_set = [item[1] for item in traindata]  # Extract all labels
            # Step 1: Extract data and labels
            x_test_set = [item[0] for item in testdata]  # Extract all first items (64x data)
            y_test_set = [item[1] for item in testdata]  # Extract all labels
            shuffle_train_dataset = ShuffleDataset(x_set, y_set, is_numpy=False)
            shuffle_test_dataset = ShuffleDataset(x_test_set, y_test_set, is_numpy=False)
            traindata = shuffle_train_dataset
            testdata = shuffle_test_dataset
        # shuffle_train_loader = DataLoader(shuffle_train_dataset, batch_size=64, shuffle=False)
        # shuffle_test_loader = DataLoader(shuffle_test_dataset, batch_size=64, shuffle=False)

    # if num_classes > model.num_classes:
    #     # When encountering new classes during incremental learning
    #     additional_neurons = num_classes - model.output_neurons  # Number of new neurons to add
    #     model.expand_fc_layer(additional_neurons)
    #     print("Expanded model:", model)

    # Train the model
    for epoch in range(epoch_number):  # Number of epochs
        print('epoch_number: ', epoch)
        if (epoch == epoch_number - 1 and sleep_param) or (epoch_number - 1 and contineual_learning):
            sleep_vgg(model=model, data=traindata, replay_iter=replay_iter_param,
                      contineual_learning=contineual_learning, task_number=task_number,
                      replay_batch_number=replay_batch_number, replay_batch_size=replay_batch_size)
            model = torch.load(
                '/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/best_model.pth',
                map_location=torch.device(device))
            # model.load_state_dict(
            #     torch.load('/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/best_model.pth',
            #                map_location=torch.device(device)))
            break
        performance_result = []
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        train_confusion_matrix = torch.zeros(10, 10)
        train_acc = 0

        for i, data in enumerate(traindata, 0):
            inputs, labels = data
            # inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # if sleep_param:
            #     activations = get_activations(model, 'features.block2_pool', inputs)
            # if i % 100 == 0 and sleep_param:
            #     sleep = True
            # else:
            #     sleep = False
            outputs = model(inputs)

            loss = criterion(outputs, labels.detach())
            _, train_predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()
            train_acc = 100 * train_correct / train_total
            for t, p in zip(labels.view(-1), train_predicted.view(-1)):
                train_confusion_matrix[t.long(), p.long()] += 1
            train_acc_per_class = train_confusion_matrix.diag() / train_confusion_matrix.sum(1)

            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss = running_loss + loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                performance_result.append({'epoch': epoch + 1, 'loss': running_loss / 100})
                running_loss = 0.0
            # Call on_epoch_end at the end of each epoch
        # if sleep_param:
        #     shuffle_train_dataset.on_epoch_end()
        #     shuffle_test_dataset.on_epoch_end()
        if train_acc < 60 and imagination_param == 'marcov':
            imagination_input = marcov_imagination(model, channel)
            for i in range(len(imagination_input)):
                traindata.append(imagination_input[i])
        if train_acc < 60 and imagination_param == 'arithmetic':
            imagination_input = arithmetic_imagination(train_acc_per_class, traindata, imagination_level)
            if imagination_input:
                traindata.append(imagination_input)
        if train_acc < 60 and imagination_param == 'ari+marc':
            imagination_input = arithmetic_imagination(train_acc_per_class, traindata, imagination_level)
            if imagination_input:
                traindata.append(imagination_input)
            imagination_input = marcov_imagination(model, channel)
            for i in range(len(imagination_input)):
                traindata.append(imagination_input[i])
        # Monitor memory usage
        # print(f"Epoch {epoch + 1}, Memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} mb")

        # Save checkpoints periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       f'/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/{model_name}/{imagination_level}/model_checkpoint_{epoch + 1}.pth')
    if epoch != 0:
        general_utils.plot_results(performance_result, 'epoch', 'loss', f'classifier_{task_number}')
    if task_number == 1000:
        # writing to csv file
        with open(
                f'/home/anahita/personal_projects/cl/my_contineual_learning/data/results/{imagination_level}/performance_result.csv',
                'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'loss'])

            # writing headers (field names)
            writer.writeheader()

            # writing data rows
            writer.writerows(performance_result)

    print('Finished Training')

    # Evaluate the model
    correct = 0
    total = 0
    results = []
    confusion_matrix = torch.zeros(10, 10)

    with torch.no_grad():
        for i, data in enumerate(testdata):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            acc_per_class = confusion_matrix.diag() / confusion_matrix.sum(1)

            results.append({'iter': i + 1, 'acc': 100 * correct / total})

    print('acc_per_class: ', acc_per_class)
    general_utils.plot_results(results, 'iter', 'acc', f'classifier_acc_{task_number}')
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    torch.save(model.state_dict(),
               f'/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/{model_name}/{imagination_level}/{dataset}_{model_name}_model.pt')

    torch.cuda.empty_cache()
    # Loading the model on CPU

    return model


def marcov_imagination(model, channel):
    # perform Monte Carlo sampling to generate new samples
    imagination_data = []
    for i in range(3):
        n_samples = 64  # number of samples to generate
        samples = torch.randn(n_samples, channel, 8, 8).to('cuda')

        outputs = model(samples)
        _, predicted = torch.max(outputs.data, 1)
        imagination_data.append([samples, predicted])
    # utils.save_image(
    #     torch.cat([dec_sample[0], predicted[0]], 0),
    #     f"data/sample/marcov_{str(0).zfill(5)}.png",
    #     nrow=n_samples,
    #     normalize=True,
    #     value_range=(-1, 1)
    # )

    return imagination_data


def arithmetic_imagination(acc_per_class, traindata, imagination_level):
    imagination_data = []
    images_arr = []
    targets_arr = []
    for i, data in enumerate(traindata, 0):
        inputs, labels = data

        for l, acc in enumerate(acc_per_class):
            if targets_arr.count(l) < 20:
                if acc < 0.6:
                    instances = inputs[(labels == l).nonzero(as_tuple=True)[0]]
                    if instances.shape[0] >= 3:
                        for n_samples in range(0, instances.shape[0] - 2):
                            if targets_arr.count(l) < 20:
                                new_data = instances[n_samples] + instances[n_samples + 1] - instances[n_samples + 2]
                                images_arr.append(new_data)
                                targets_arr.append(l)
                    elif instances.shape[0] == 2:
                        new_data = (instances[0] + instances[1]) / 2
                        images_arr.append(new_data)
                        targets_arr.append(l)
                    elif instances.shape[0] == 1:
                        new_data = (instances[0] + instances[0]) / 2
                        images_arr.append(new_data)
                        targets_arr.append(l)
    if len(images_arr) != 0:
        for la in range(10):
            print(f'len_arithmetic_data: {targets_arr.count(la)}, label: {la}')
        images = torch.stack(images_arr)
        targets = torch.as_tensor(targets_arr)
        # plot_imagination(images, targets, 'arithmetic', imagination_level)
        return [images, targets]
    else:
        return False


def plot_imagination(images, targets, im_type, imagination_level):
    if imagination_level == 'data':
        # for i, img in enumerate(images):
        general_utils.plot_images(images, targets, im_type, imagination_level)

    elif imagination_level == 'enc_b1':
        g_model = VQVAE().to(device)
        g_model.load_state_dict(torch.load("checkpoint/vqvae_200.pt"))
        decoded_imaginations = []
        # for i, img in enumerate(images):
        with torch.no_grad():
            qa = g_model.encoder_to_quant(images.to(device))
            decoded_images = g_model.decode(qa[0], qa[1])
            decoded_imaginations.append(
                [decoded_images, targets])
        general_utils.plot_images(decoded_imaginations[0][0], decoded_imaginations[0][1], im_type, imagination_level)


def delete_files(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # List the contents of the directory
        contents = os.listdir(directory)

        # Check if the directory is empty
        if not contents:
            print(f"The directory {directory} is empty. No files or folders to delete.")
        else:
            # Loop through the directory and remove all files and folders
            for filename in contents:
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the folder and its contents
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"All files and folders in {directory} have been deleted.")
    else:
        print(f"The directory {directory} does not exist.")


def activation(model, data, activation_layer, sleep):
    inputs = data.x_shuf
    labels = data.y_shuf
    test_activation = []
    basedir = '/home/anahita/personal_projects/cl/my_contineual_learning/data/activations'

    delete_files(basedir)

    for input_data in inputs:
        test_activation.append(get_activations(model, activation_layer, input_data))
    # print('activations.shape: ', test_activation.shape)

    # test_activation_values = test_activation.values()
    # data = list(test_activation_values)
    # data_array = np.array(test_activation)
    # data_array_squeezed = np.squeeze(data_array)
    # filtered_tensor_list = [t for t in test_activation if t.size(0) == 64]

    # data_array_squeezed = torch.stack(filtered_tensor_list)
    # saved as float16 to save disk space
    # data_array_squeezed = data_array_squeezed.astype('float16')
    # save these activations
    class_list = []
    for i, avtivation_data in enumerate(test_activation):
        # activation_dir = os.path.join(basedir, ''.join((labels[i].cpu().numpy().astype(str).flatten())))
        activation_dir = basedir
        if not os.path.exists(activation_dir):
            os.makedirs(activation_dir)
        os.chdir(activation_dir)
        extension = '.pth'
        data_title = ''.join((labels[i].cpu().numpy().astype(str).flatten())) + extension
        if len(''.join((labels[i].cpu().numpy().astype(str).flatten()))) == 64:
            torch.save(avtivation_data, f'{activation_dir}/{data_title}')
            class_list.append(data_title)
    with open('/home/anahita/personal_projects/cl/my_contineual_learning/data/class_list.txt', 'w') as file:
        for item in class_list:
            file.write(item + '\n')
    # return class_list
    # np.save(data_title, data_array_squeezed)


def get_distributions_single_img(class_list, vgg_model):
    activation_path_full = '/home/anahita/personal_projects/cl/my_contineual_learning/data/activations'
    list_of_files = sorted(os.listdir(activation_path_full))
    classes = []
    len_y = len(list_of_files) * 64
    test_data_array = []
    for i, image_classes in enumerate(class_list):
        temp_data = torch.load(os.path.join(activation_path_full, list_of_files[i])).cpu().numpy()
        # temp_data = np.expand_dims(temp_data, axis=0)
        temp_data_arrays = [temp_data[i] for i in range(temp_data.shape[0])]
        len_x1 = int((temp_data.shape[1]) / 4)
        len_x2 = int((temp_data.shape[2]) / 2)
        len_x3 = temp_data.shape[3]
        len_x = len_x1 * len_x2 * len_x3
        all_data = np.zeros((len_x, 1))

        for j, image_class in enumerate(image_classes[:-4]):

            new_filter = np.zeros((len_x1, len_x2, len_x3))
            for n in range(len_x3):
                temp_filter_full = temp_data_arrays[j][:, :, n]
                temp_filter_downsampled = np.mean(temp_filter_full, axis=1)
                temp_filter_downsampled = np.reshape(temp_filter_downsampled, (len_x1, 2, len_x2, 2)).mean(axis=(1, 3))
                new_filter[:, :, n] = temp_filter_downsampled
            downsampled_data = new_filter.reshape(len_x1, 2)
            # Convert the NumPy array to a PyTorch tensor
            downsampled_data_tensor = torch.from_numpy(downsampled_data).float().to(device)
            test_sample = generate_test_sample(downsampled_data_tensor, 1)
            for n in range(1):
                temp_sample = test_sample[:]
                temp_sample_reshaped = np.reshape(temp_sample, (len_x1 // 2, len_x2, len_x3))
                new_upsampled_array = np.zeros((len_x1 * 8, len_x2 * 2, len_x3))
                for m in range(len_x3):
                    temp_filter = temp_sample_reshaped[:, :, m]
                    temp_filter_upsampled = np.repeat(np.repeat(temp_filter, 16, axis=0), 2, axis=1)
                    new_upsampled_array[:, :, m] = temp_filter_upsampled
                new_upsampled_array = new_upsampled_array.astype('float16')
                upsampled_distribution_data_path_full = '/home/anahita/personal_projects/cl/my_contineual_learning/data/unsampled_distribution_single_img/'
                sample_filename = str(i) + '_' + str(j) + '_' + str(n) + '_' + image_class + '.npy'
                np.save(os.path.join(upsampled_distribution_data_path_full, sample_filename), new_upsampled_array)


def get_distributions(class_list, vgg_model, batch_number, num_sample_size, contineual_learning):
    # class_list = np.load('/home/daniel/replay/class_lists/' + str(vgg_model) + '.npy')
    if not contineual_learning:
        delete_files('/home/anahita/personal_projects/cl/my_contineual_learning/data/unsampled_distribution')

    for image_class in class_list[0:batch_number]:
        activation_path_full = '/home/anahita/personal_projects/cl/my_contineual_learning/data/activations'
        list_of_files = sorted(os.listdir(activation_path_full))

        temp_data = torch.load(os.path.join(activation_path_full, list_of_files[0])).cpu().numpy()
        temp_data = np.expand_dims(temp_data, axis=0)

        for files in list_of_files[1:]:
            temp_data_2 = torch.load(os.path.join(activation_path_full, files)).cpu().numpy()
            temp_data_2 = np.expand_dims(temp_data_2, axis=0)
            temp_data = np.append(temp_data, temp_data_2, axis=0)

        # len_x1 = int((len(np.ndarray.flatten((temp_data[0, :, 0, 0])))) / 2)
        # len_x2 = int((len(np.ndarray.flatten((temp_data[0, 0, :, 0])))) / 2)
        # len_x3 = int(len(np.ndarray.flatten((temp_data[0, 0, 0, :]))))
        len_x1 = int((temp_data.shape[1]) / 4)
        len_x2 = int((temp_data.shape[2]) / 4)
        len_x3 = temp_data.shape[3]
        len_x = len_x1 * len_x2 * len_x3
        len_y = len(list_of_files)

        all_data = np.zeros((len_x, len_y))

        for images in range(len_y):
            new_filter = np.zeros((len_x1, len_x2, len_x3))
            for n in range(len_x3):
                temp_filter_full = temp_data[images, :, :, n]
                # temp_filter_full = np.reshape(temp_filter_full, (len_x1, 4, len_x1, 4))
                # temp_filter_downsampled = np.mean(np.mean(temp_filter_full, axis=3), axis=2)
                temp_filter_downsampled = np.mean(temp_filter_full, axis=-1)
                temp_filter_downsampled = temp_filter_downsampled.reshape(len_x1, 4, len_x2, 4).mean(axis=(1, 3))
                new_filter[:, :, n] = temp_filter_downsampled
            all_data[:, images] = new_filter.flatten()
        # Convert the NumPy array to a PyTorch tensor
        all_data_tensor = torch.from_numpy(all_data).float().to(device)
        test_sample = generate_test_sample(all_data_tensor, num_sample_size)

        for n in range(num_sample_size):
            temp_sample = test_sample[n, :]
            temp_sample_reshaped = np.reshape(temp_sample, (len_x1, len_x2, len_x3))
            new_upsampled_array = np.zeros((len_x1 * 4, len_x2 * 4, len_x3))
            for m in range(len_x3):
                temp_filter = temp_sample_reshaped[:, :, m]
                temp_filter_upsampled = np.repeat(np.repeat(temp_filter, 4, axis=0), 4, axis=1)
                new_upsampled_array[:, :, m] = temp_filter_upsampled

            # new_upsampled_array = np.repeat(new_upsampled_array, 2, axis=2)
            new_upsampled_array = new_upsampled_array.astype('float16')
            upsampled_distribution_data_path_full = os.path.join(
                '/home/anahita/personal_projects/cl/my_contineual_learning/data/unsampled_distribution', image_class)
            os.makedirs(upsampled_distribution_data_path_full, exist_ok=True)
            sample_filename = str(n) + '.npy'
            np.save(os.path.join(upsampled_distribution_data_path_full, sample_filename), new_upsampled_array)


class ActivationHook:
    def __init__(self, activations, layer_name):
        self.activations = activations
        self.layer_name = layer_name

    def __call__(self, module, input_data, output):
        self.activations[self.layer_name] = output.detach()


def get_activations(model, layer_name, input_data):
    activations = {}

    # Register the hook
    for name, module in model.named_modules():
        if name == layer_name:
            hook = ActivationHook(activations, layer_name)
            module.register_forward_hook(hook)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Run a forward pass to get the activations
    model(input_data)

    return activations[layer_name]


def sleep_vgg(model, data, replay_iter, contineual_learning, task_number, replay_batch_number, replay_batch_size):
    with open('/home/anahita/personal_projects/cl/my_contineual_learning/data/class_list.txt', 'r') as file:
        class_list = [line.strip() for line in file]
    current_model_save_filepath_full = '/home/anahita/personal_projects/cl/my_contineual_learning'
    activation(model, data, 'features.block1_pool', contineual_learning)
    get_distributions(class_list, model, replay_batch_number, replay_batch_size, contineual_learning)
    # get_distributions_single_img(class_list, model)
    replay_vgg(current_model_save_filepath_full, model, class_list, distribution_type='batch', replay_iter=replay_iter,
               task_number=task_number)


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


def load_npy_files(main_folder):
    data_samples = {}  # Dictionary to store arrays with their file paths as keys

    # Traverse the main folder and its subfolders
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                # Load the .npy file as a numpy array
                data = np.load(file_path)
                # Store the array in the dictionary with the file path as the key
                data_samples[file_path] = data

    return data_samples


def replay_vgg(current_model_save_filepath_full, vgg_model, class_list, distribution_type, replay_iter, task_number):
    train_dir = '/home/anahita/personal_projects/cl/my_contineual_learning/data/unsampled_distribution'
    # Get training files and labels
    train_sets = []
    for dp, dn, fn in os.walk(train_dir):
        dn[:] = [d for d in dn]
        for f in fn:
            if distribution_type == 'image':
                train_sets.append((os.path.join(dp, f), f.split('_')[-1].split('.')[0]))
            if distribution_type == 'batch':
                train_sets.append((os.path.join(dp, f), dp.split('/')[-1].split('.')[0]))

    # Separate file paths and class labels

    x_paths, y_cls_train = zip(*train_sets)

    # Load data from the file paths into x_train
    x_train = [np.load(path) for path in x_paths]  # Load each .npy file

    # Convert y_cls_train to a list (if not already)
    y_train = [list(label.replace('.pth', '')) for label in y_cls_train]
    # Example of converting string labels to integers

    # y_classes = {class_list[i]: i for i in range(10)}
    # y_train = [y_classes[y] for y in y_cls_train]
    # y_train_oh = to_categorical(np.copy(y_train), num_classes=10)

    # Get validation files and labels
    validation_dir = '/home/anahita/personal_projects/cl/my_contineual_learning/data/activations'
    val_sets = []
    for dp, dn, fn in os.walk(validation_dir):
        dn[:] = [d for d in dn if d in class_list]
        for f in fn:
            val_sets.append((os.path.join(dp, f), f.split('_')[-1].split('.')[0]))

    x_val, y_cls_val = zip(*val_sets)
    # y_val = [y_classes[y] for y in y_cls_val]
    # y_val_oh = to_categorical(np.copy(y_val), num_classes=10)

    # Data loaders
    # train_dataset = ShuffleDataset(x_train, y_train, is_numpy=True)
    val_dataset = ShuffleDataset(x_val, y_cls_val, is_numpy=True)

    # train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=26, shuffle=False)

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Load the model
    # model = torch.load(current_model_save_filepath_full)
    vgg_model.eval()

    # Build new model with the input shape matching the replay representations
    class NewModel(nn.Module):
        def __init__(self, base_model, start_idx):
            super(NewModel, self).__init__()
            base_model.features = nn.Sequential(*list(vgg_model.children()))[0][start_idx:]
            # base_model.classifier = nn.Sequential(*list(*list(vgg_model.children())[1])[:])

        def forward(self, x):
            return self.base_model(x)

    start_idx = 5
    start_block = 1
    input_shape = vgg_model.features[start_idx].in_channels
    new_model = copy.deepcopy(vgg_model)
    NewModel(new_model, start_idx).to('cpu')

    # Compile the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(new_model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0)
    performance_result = []
    # Train the replay model for one epoch
    new_model.train()
    for epoch in range(replay_iter):
        # for images, labels in (x_train,y_train):
        #     images = images.to('cuda')
        #     labels = torch.argmax(torch.tensor(labels), dim=1).to('cuda')
        #     optimizer.zero_grad()
        #     outputs = new_model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            reduced_data = torch.mean(x_batch, dim=1, keepdim=True)
            reshaped_data = reduced_data.repeat(1, 4, 1, 1)
            input_reshaped = reshaped_data.permute(0, 2, 3, 1)
            outputs = new_model(input_reshaped.to('cuda'))

            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'New model replay  Loss: {running_loss / len(train_loader)}')
        performance_result.append({'epoch': epoch + 1, 'loss': running_loss / len(train_loader)})

    general_utils.plot_results(performance_result, 'epoch', 'loss', f'replay_classifier_{task_number}')

    # Save the new model weights
    torch.save(new_model.state_dict(),
               '/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/vgg_replay_model.pth')

    # Load the new weights back into the main model
    vgg_model.load_state_dict(
        torch.load('/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/vgg_replay_model.pth'),
        strict=False)

    # Save the updated model
    torch.save(vgg_model, '/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/vgg_model.pth')

    # Compare current loss with previous loss and save the best model
    current_loss = loss.item()  # assuming validation loss is calculated
    checkpoint_directory_full = '/home/anahita/personal_projects/cl/my_contineual_learning/checkpoint/vgg/'
    current_loss_filepath = os.path.join(checkpoint_directory_full, 'lowest_loss.npy')

    if os.path.isfile(current_loss_filepath):
        previous_lowest_loss = np.load(current_loss_filepath)
        if current_loss < previous_lowest_loss:
            np.save(current_loss_filepath, current_loss)
            best_model_filepath = os.path.join(checkpoint_directory_full, 'best_model.pth')
            torch.save(vgg_model, best_model_filepath)
    else:
        np.save(current_loss_filepath, current_loss)
        best_model_filepath = os.path.join(checkpoint_directory_full, 'best_model.pth')
        torch.save(vgg_model, best_model_filepath)
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()


def generate_test_sample(tensor_data, size):
    # Transpose the tensor
    #  tensor_data = tensor_data.T
    # Step 1: Compute the mean of each column
    mean = torch.mean(tensor_data.T, dim=0).cpu().numpy()
    covariance = torch.cov(tensor_data).cpu().numpy()

    covariance = (covariance + covariance.T) / 2

    # rowvar=False treats rows as observations
    # # Step 2: Subtract the mean from the matrix
    # matrix_centered = tensor_data.cpu().numpy() - mean
    #
    # # Step 3: Compute the covariance matrix
    # cov_matrix = torch.matmul(matrix_centered.T(), matrix_centered) / (tensor_data.size(0) - 1)
    # cov_matrix = np.load('/home/anahita/personal_projects/cl/my_contineual_learning/data/numpy_cov_matrix.npy')
    # mean = np.load('/home/anahita/personal_projects/cl/my_contineual_learning/data/mean.npy')
    fixed_covariance = nearest_positive_definite(covariance)

    # cov_matrix = np.dot(matrix_centered.T, matrix_centered) / (tensor_data.shape[0] - 1)
    test_sample = multivariate_normal.rvs(mean, fixed_covariance, size=size)
    # print("Mean:\n", mean)
    # print("Covariance:\n", covariance)

    return test_sample


import numpy as np


def nearest_positive_definite(matrix):
    """Find the nearest positive definite matrix to the input."""
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(matrix))
    # The matrix is not positive definite, adjusting it
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(A3):
        A3 += I * spacing * k
        k += 1
    return A3


def is_positive_definite(matrix):
    """Returns True if the matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        flattened_y_data = [label for batch in y_data for label in batch]

        # Optionally create a mapping if labels are not already integers
        label_mapping = {label: idx for idx, label in enumerate(set(flattened_y_data))}
        self.y_data = torch.tensor([label_mapping[label] for label in flattened_y_data], dtype=torch.long).to('cpu')

        self.x_data = [torch.tensor(x, dtype=torch.float32).to('cpu') for x in x_data]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
