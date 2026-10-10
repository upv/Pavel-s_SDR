import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# TODO: Exclude following libs in next release:
import sionna as sn
import tensorflow as tf

def create_preamble(N_FFT:int, CP_len:int, N_repeat:int = 2):
    """ Creates Preamble for synchronization and carrier frequency compensation
    Uses Schmidl's OFDM preamble design scheme

    TODO: from https://liusc1028.github.io/papers/ISPLC13.pdf
    implement Minn's and Proposed methods 

    Args:
        N_FFT (int): defines length of the preamble
        CP_len (int): cyclic prefix length added to preamble
        N_repeat (int, optional): Number of preamble repetitions. Defaults to 2.

    Returns:
        _type_: _description_
    """
    # random BPSK symbols
    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_FFT/N_repeat), 1))
    preamble = np.complex64(preamble)
    # repeat preamble
    preamble_full = np.tile(preamble, (N_repeat, 1))
    # add cyclic prefix
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    return preamble_full_cp

def ZadoffChu_gen(n_US, M_sc, ID_Ue):
    '''Generates a Zadoff-Chu sequence used for cell search and synchronization in LTE networks.

    Inputs:
       n_US  - The number of unique sequences.
       M_sc  - The number of subcarriers.
       ID_Ue - The unique identifier for the user equipment (UE).

    Output:
       out   - The generated Zadoff-Chu sequence.
    '''

    # Configuration for Zadoff-Chu sequence generation
    N_ap = 1                                                    # Number of antenna ports

    # Length of Zadoff-Chu sequence, chosen as the max prime less than M_sc
    N_zc = max([p for p in range(2, M_sc) if all(p % n != 0 for n in range(2, p))])
    n = np.arange(N_zc)                                         # Base sequence index

    # Generate the base Zadoff-Chu sequence
    q = 10                                                      # Base root index (should be relatively prime to N_zc)
    x_q = np.exp(-2j * np.pi * q * n * (n + 1) / (2 * N_zc))
    k = np.arange(M_sc)                                         # Subcarrier indices
    r = x_q[np.mod(k, N_zc)]                                    # The sequence repeated for all subcarriers

    # Calculate the cyclic shift for the sequence
    n_rs = ID_Ue                                                # UE-specific sequence number
    p = np.arange(N_ap)                                         # Antenna port indices
    N_rs = np.mod(n_rs + n_US * p / N_ap, n_US)                 # Cyclic shift calculation
    alpha = 2 * np.pi * N_rs / n_US                             # Cyclic shift phase
    n = np.arange(len(r))                                       # Index for the final sequence

    # Initialize the output sequence matrix
    out = np.zeros((M_sc, N_ap), dtype = complex)

    # Apply the cyclic shift to generate the Zadoff-Chu sequence for each antenna port
    for index_antenna in range(len(alpha)):
        out[:, index_antenna] = np.exp(1j * alpha[index_antenna] * n) * r

    return out


def generate_bits(N_subcarriers:int, N_bits_per_conts_point:int):
    """generates random sequence of bits with defined size

    Args:
        N_subcarriers (int): Number of subcarriers used for data transmission
        N_bits_per_conts_point (int): Number of bits in QAM symbol

    Returns:
        ndarray: returns bits 
    """
    data_stream = np.random.binomial(n = 1, p = 0.5, size = (N_subcarriers*N_bits_per_conts_point))
    return data_stream

def qam_modulate(bits:np.ndarray, num_bits_per_symbol:int):
    """Applies QAM modulation for provided bitstream
    TODO Temporary algorithm! Reimplement same modem with numpy logic.

    Args:
        bits (ndarray): array of bits with shape of [1 x N_symbols*num_bits_per_symbol]
        num_bits_per_symbol (int): number of bits per QAM symbol

    Returns:
        ndarray: Normalized QAM symbols with shape of [1 x N_symbols]
    """
    const_type = "pam" if num_bits_per_symbol == 1 else "qam"
        
    constellation = sn.mapping.Constellation(const_type, num_bits_per_symbol)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(tf.convert_to_tensor(bits[None,...])).numpy()

def qam_demodulate(signal:np.ndarray, num_bits_per_symbol:int):
    """ QAM demodulator with soft decision. 
    TODO Temporary algorithm! Reimplement same modem with numpy logic.

    Args:
        signal (ndarray): array with QAM symbols with shape of [1 x N_symbols]
        num_bits_per_symbol (int): number of bits per QAM symbol

    Returns:
        ndarray: demodulated bitstream with shape of [1 x N_symbols*num_bits_per_symbol]
    """
    const_type = "pam" if num_bits_per_symbol == 1 else "qam"
    constellation = sn.mapping.Constellation(const_type, num_bits_per_symbol)
    demapper = sn.mapping.Demapper("app", constellation = constellation)
    return (demapper([tf.convert_to_tensor(signal), 0.0]).numpy() > 0).astype(np.int64)

def form_baseband_ofdm_symb(qam_symb:np.ndarray, N_FFT:int, CP_len:int, dc_offset:bool = False):
    """Form timedomain baseband OFDM symbol using input QAM symbols 

    Args:
        qam_symb (ndarray): array of QAM symbols
        N_FFT (int): Number of FFT samples
        CP_len (int): Cyclic prefix length
        dc_offset (bool, optional): Whether use DC component for allocating data. Defaults to False.

    Returns:
        ndarray: time-domain baseband OFDM symbol with cyclic prefix
    """
    qam_symb = qam_symb.ravel()
    N_subcarriers = qam_symb.shape[0]
    assert N_subcarriers < N_FFT, 'Cannot send more symbols than FFT size'
    dc = int(dc_offset)
    
    # Frequency-domain baseband OFDM symbol
    tx_ofdm_symb = np.zeros((N_FFT), dtype = np.complex64)
    tx_ofdm_symb[dc : N_subcarriers//2 + dc] = qam_symb[N_subcarriers//2:]
    tx_ofdm_symb[-N_subcarriers//2:] = qam_symb[:N_subcarriers//2]

    # Time-domain baseband OFDM symbol 
    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_symb, axis = 0, norm = 'ortho')
    time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )
    return time_ofdm_sym_cp_pilot


def create_data(N_sc, N_fft, CP_len, dc_offset = False, aditional_return = None):
    
    data_stream =  np.random.randint(0, 2, size=(N_sc, 1))
    data_stream_mod = 1 - 2 * data_stream
    mod_sym_pilot = np.complex64(data_stream_mod)

    tx_ofdm_sym = np.zeros((N_fft, 1), dtype = np.complex64)
    dc = int(dc_offset)
    tx_ofdm_sym[dc : N_sc//2 + dc] = mod_sym_pilot[ N_sc//2: ]
    tx_ofdm_sym[-N_sc//2: ] = mod_sym_pilot[ 0 : N_sc//2]

    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, axis = 0, norm = 'ortho')
    time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )

    if aditional_return == 'mod_sym_pilot':
        return time_ofdm_sym_cp_pilot, mod_sym_pilot
    if aditional_return == 'data_stream':
        return time_ofdm_sym_cp_pilot, data_stream

#############################################################################
def find_edges(rx_sig:np.ndarray, frame_len:int, preamble_len:int, CP_len:int, start_idx = 0):
    """Seeks index with beginning of the frame using structure of preamble and carrier frequency offset argument

    Args:
        rx_sig (ndarray): received signal samples in the time domain
        frame_len (int): Length of the frame
        preamble_len (int): Length of the UNIQUE part of the preamble signal.
                               If preamble's structure is [cp,A,A], then preamble_len is len of [A]
        CP_len (int): Length of the preamble cyclic prefix
        start_idx (int): The first index from which searching is started

    Returns:
        idx_max_filt (int): index of correlation maximum, representing start of the frame 
        corr_coeff (complex): complex-valued coefficient of the correlation maximum, which will be used for 
                              carrier frequency offset compensation. 
    """
    corr_list = []  # correlation results
    for idx_search in range(start_idx, start_idx + frame_len):                          # Seeking start of frame in the len of frame signal
        first_part = rx_sig[idx_search: idx_search + preamble_len]                      # Extract first part with len of preamble 
        second_part = rx_sig[idx_search + preamble_len: idx_search + 2 * preamble_len]  # Extract second part (shifted by preamble_len) with len of preamble
        corr_list.append( np.dot(np.conj(first_part), second_part) )                    # Add dot product of two parts to corr_list                                   

    corr_list_abs = np.abs(np.array(corr_list))                                         # Magnitude of correlation function
    
    # filtering is applied since preamble correlation function has plateau 
    filter_cp = np.ones(CP_len) / np.sqrt(CP_len)                                       # Small mean filter
    corr_list_filt = np.convolve(filter_cp, corr_list_abs, mode='same')                 # Apply filtering

    rel_idx_filt = np.argmax(corr_list_filt, axis=0)                                    # find position of peak 
    idx_max_filt = rel_idx_filt + start_idx                                             # Add starting index
    idx_max_filt = idx_max_filt + int(CP_len/2)                                         # Add half of the CP. TODO: Why? Test it 

    return idx_max_filt, corr_list[rel_idx_filt]

def cfo_compensate(frame_receive:np.ndarray, corr_value:complex, preamble_len:int):
    """ Reduces carrier frequency offset from the received samples

    Args:
        frame_receive (np.ndarray): Received frame samples
        corr_value (complex): Value of correlation function obtained from @find_edges func
        preamble_len (int): length of the part of preamble 

    Returns:
        np.ndarray: frame samples with compensated cfo
    """
    angle_cfo = np.angle(corr_value) / preamble_len                                     # calculate phase-shift  
    cfo_comp_sig = np.exp(1j * (-angle_cfo * np.arange(0, len(frame_receive))) )        # form phase rotation vector
    frame_receive = frame_receive * cfo_comp_sig                                        # apply phase rotation
    return frame_receive

def baseband_freq_domian(ofdm_symb_freq:np.ndarray, N_sc_use:int, dc_offset:bool = False):
    """Extract QAM symbols from Frequency domain ofdm signal

    Args:
        ofdm_symb_freq (np.ndarray): Frequency domain ofdm symbol
        N_sc_use (int): Number of QAM symbols to be extracted
        dc_offset (bool, optional): Whether to try extract QAM symbol from DC component. Defaults to False.

    Returns:
        np.ndarray: Array with extracted QAM symbols
    """
    dc_offset = int(dc_offset)
    rec_sym_pilot = np.zeros((N_sc_use,1), dtype = np.complex64)
    rec_sym_pilot[int(N_sc_use/2):, 0] = ofdm_symb_freq[0+dc_offset:int(N_sc_use/2) + dc_offset]
    rec_sym_pilot[0:int(N_sc_use/2), 0] = ofdm_symb_freq[-int(N_sc_use/2):]
    
    return rec_sym_pilot

def svd_combining(rx_data:np.ndarray) -> np.ndarray:
    """ Applies svd combining for coherent reception

    Args:
        rx_data (np.ndarray): received samples with shape of [N_rx_ants, X]
        where N_rx_ants is number of antennas at the receiver

    Returns:
        np.ndarray: rx samples with shape of [1 , X]
    """
    R = rx_data @ rx_data.conj().T           # covariance matrix 
    U,s,_ = np.linalg.svd(R)                 # extract left basis
    u = U[:,0]                               # use 1st singular vector
    combined = u.conj().T @ rx_data          # combining
    return combined

def channel_estimation(h_ls:np.ndarray, CP_len:int, N_fft:int):
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')          # transform channel into delay domain
    ce_len = len(h_ls)               

    # define edges for filter 
    W_spead = int(CP_len/2)
    W_sync_err = int(CP_len)
    W_max = W_spead + W_sync_err
    W_min = W_sync_err

    # construct filter
    eta_denoise = np.zeros_like(h_time)
    eta_denoise[-W_min:] = 1.0 
    eta_denoise[0:W_max] = 1.0

    # apply filter
    h_time_denoise = h_time * eta_denoise

    # transform signal back to the frequency domain
    h_hw = np.fft.fft(h_time_denoise, N_fft, 0, norm='ortho')
    h_ls = h_hw[0:ce_len]

    # display.clear_output(wait = True)
    # plt.figure(100)
    # plt.plot(np.arange(0, N_fft), np.abs(h_time))
    # plt.plot(np.arange(0, N_fft), np.abs(h_time_denoise))
    # plt.title('Channel response time domain')
    # plt.grid(); plt.show()

    return h_ls

def estimate_SNR(pilot_freq,rec_sym_pilot,N_sc_use):
    noise_arr = pilot_freq[int(N_sc_use/2) : -int(N_sc_use/2)]
    sigma0_freq = np.real( np.dot( np.conj(noise_arr) , noise_arr ) ) / len(noise_arr)
    Es_freq = np.real( np.dot( np.conj(rec_sym_pilot[:,0]) , rec_sym_pilot[:,0] ) ) / len(rec_sym_pilot[:, 0])
    SNR_est = 10.0 * np.log10(Es_freq / sigma0_freq)
    return SNR_est

def demodulate(eq_data, N_sc_use):
    bit_arr = []    
    for idx in range(0, N_sc_use):
        
        if (np.real(eq_data[idx]) > 0):
            bit_curr = 0
        else:
            bit_curr = 1
            
        bit_arr.append(bit_curr)
    return bit_arr

def get_ber(data_stream, bit_arr, N_sc_use):
    err_num = 0
    for idx in range(0, N_sc_use):
        if (data_stream[idx, 0] != bit_arr[idx]):
            err_num = err_num + 1
    ber = 1.0 * err_num / N_sc_use
    return ber

#############################################################################
