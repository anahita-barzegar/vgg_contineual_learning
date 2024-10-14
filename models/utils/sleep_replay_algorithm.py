import numpy as np


def sleep_replay_algorithm(nn, num_iterations, opts, sleep_opts, sleep_input, tx, ty):
    dt = opts['dt']
    errors = []

    num_features = sleep_input.shape[0]
    sleep_opts['DC'] = 0

    W_old = nn.W.copy()

    for t in range(num_iterations):
        rescale_fac = 1 / (dt * opts['max_rate'])
        spike_snapshot = np.random.rand(num_features, 1) * rescale_fac / 2
        inp_image = spike_snapshot <= sleep_input[:, t]

        nn.layers[0].spikes = inp_image
        nn.layers[0].sum_spikes += inp_image
        nn.layers[0].total_spikes[t, :] = nn.layers[0].spikes

        for l in range(1, len(nn.size)):
            impulse = sleep_opts['alpha'][l - 1] * np.dot(nn.layers[l - 1].spikes.T, nn.W[l - 1])

            impulse -= np.sum(impulse) / len(impulse) * sleep_opts['W_inh']

            nn.layers[l].mem = sleep_opts['decay'] * nn.layers[l].mem + impulse.T

            if l == 3:
                nn.layers[l].mem += sleep_opts['DC']

            nn.layers[l].spikes = nn.layers[l].mem >= opts['threshold'] * sleep_opts['beta'][l - 1]

            post = nn.layers[l].spikes
            pre = nn.layers[l - 1].spikes

            nn.W[l - 1][post, pre] += sleep_opts['inc'] * sigmoid(nn.W[l - 1][post, pre])
            nn.W[l - 1][post, ~pre] -= sleep_opts['dec'] * sigmoid(nn.W[l - 1][post, ~pre])

            nn.layers[l].mem[nn.layers[l].spikes] = 0
            nn.layers[l].refrac_end[nn.layers[l].spikes] = t + opts['t_ref']
            nn.layers[l].sum_spikes += nn.layers[l].spikes
            nn.layers[l].total_spikes[t, :] = nn.layers[l].spikes

        if (t - 1) % 20 == 0:
            er, _ = nntest(nn, tx, ty)
            errors.append((1 - er) * 100)

    if sleep_opts['normW'] == 1:
        for l in range(1, len(nn.size)):
            nn.W[l - 1] = sleep_opts['gamma'] * nn.W[l - 1] / (np.max(nn.W[l - 1]) - np.min(nn.W[l - 1])) * \
                          (np.max(W_old[l - 1]) - np.min(W_old[l - 1]))

    return nn, errors


def sigmoid(x):
    return 2 * (1.0 - (1.0 / (1 + np.exp(-x / 0.001))))
