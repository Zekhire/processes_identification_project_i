import scipy.io.wavfile as wave
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import time


def load_wav(input_path):
    audio = wave.read(input_path)
    samples = audio[1]
    Fs = audio[0]
    samples = scaling_down(samples)
    return samples, Fs


def scaling_down(x):
    x_scaled = x/float(2**15)
    return x_scaled


def save_audios(output_path, samples, Fs):
    # samples = scaling_down(samples)
    wave.write(output_path, Fs, samples)
    

def project_i(y, r, debug=False):
    len_y = len(y)
    # len_samples = 1

    c          = 5000
    theta_dash = np.transpose(np.matrix(np.zeros(r)))
    fi         = np.transpose(np.matrix(np.zeros(r)))
    L          = np.matrix(np.diag(np.ones(r)))
    # L          = np.matrix(np.diag(np.array([0.964, 0.949, 0.949, 0.975])))   # very fast stabilization? xd
    epsilon    = 0
    P_wave     = np.matrix(np.diag(c*np.ones(r)))
    y_dash     = 0
    M          = 4
    je         = np.zeros(M)
    mi         = 3

    y_asterisk = np.zeros(len(y))

    for i in range(len(y)):
        # EWLS
        print(np.squeeze(np.array(np.transpose(theta_dash))))

        if i > 0:
            fi    = np.matrix(np.roll(np.array(fi), 1))
            fi[0] = y[i-1]

        # Prediction
        y_dash_i_pipe_i_sub_1 = np.array(np.transpose(fi)*theta_dash)[0][0]    # last y sample in shifted fi and non updated theta_dash

        # Noise impulse detection
        je_i = y[i] - y_dash_i_pipe_i_sub_1
        sigma_ie_dash_sqr = np.sum(je)/M                   # previous
        # print(abs(je_i), mi*sigma_ie_dash_sqr)

        if abs(je_i) <= mi*np.sqrt(sigma_ie_dash_sqr):
            if debug:
                print("brak impulsu :(")
            d = 0                                       # noise impulse absent
            y_asterisk[i] = y[i]
        else:
            if debug:
                print("impuls :)")
            d = 1                                       # noise impulse present
            y_k_sub_1 = 0
            y_k_add_1 = 0
            if i-1 >= 0:
                y_k_sub_1 = y[i-1]
            if i+1 < len(y):
                y_k_sub_1 = y[i+1]
            y_asterisk[i] = (y_k_sub_1 + y_k_add_1)/2

        je    = np.roll(je, 1)
        je[0] = je_i**2

        y[i] = y_asterisk[i]
        epsilon    = y[i] - np.array(np.transpose(fi)*theta_dash)[0][0]
        k          = P_wave*fi/(1 + np.transpose(fi)*P_wave*fi)
        theta_dash = theta_dash + k*epsilon
        P_wave     = L*(P_wave - (P_wave*fi*np.transpose(fi)*P_wave)/(1 + (np.transpose(fi)*P_wave*fi)))*L

    print(np.squeeze(np.array(np.transpose(theta_dash))))
    return y_asterisk




if __name__ == "__main__":
    # Editable parameters
    r = 4

    samples, Fs = load_wav("src/01.wav")
    samples_asterisk = project_i(samples, r)
    save_audios("src_asterisk/01.wav", samples_asterisk, Fs)