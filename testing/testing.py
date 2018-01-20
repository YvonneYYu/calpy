import numpy
from .. import rqa
from .. import dsp
from .. import utilities
import os
import scipy.io.wavfile
from . import experiments

def test_dsp(kind="small"):
    
    if kind=="small":
        N = 1

    data_file_name  = os.path.join(os.path.dirname(__file__), 'test_dsp_profile_'+str(N)+'ms.npz')
    sound_file_name = os.path.join(os.path.dirname(__file__), 'ms_'+str(N)+'.wav')

    data = numpy.load(data_file_name)

    time_step, frame_window, eps = data['time_step'], data['frame_window'], data['eps']
    padded_signal = data['padded_signal']
    dB_profile = data['dB_profile']
    pause_profile, pause_profile_aligned = data['pause_profile'], data['pause_profile_aligned']
    pitch_profile = data['pitch_profile']

    sampling_rate, sound = scipy.io.wavfile.read(sound_file_name)

    my_padded_signal = utilities.pad_signal( sound, sampling_rate, time_step, frame_window )
    print("Padded signal : {}".format("pass" if numpy.array_equal(my_padded_signal, padded_signal) else "FAIL"))

    my_dB_profile = dsp.dB_profile( my_padded_signal, sampling_rate )
    print("dB profile : {}".format("pass" if numpy.array_equal(my_dB_profile, dB_profile) else "FAIL"))

    my_pause_profile = dsp.pause_profile( my_padded_signal, sampling_rate )
    print("pause profile : {}".format("pass" if numpy.array_equal(my_pause_profile, pause_profile) else "FAIL"))

    my_aligned_pause_profile = utilities.align_pause_to_time( my_pause_profile, sampling_rate )
    print("aligned pause profile : {}".format("pass" if numpy.array_equal(my_aligned_pause_profile, pause_profile_aligned) else "FAIL"))

    my_pitch_profile = dsp.pitch_profile( my_padded_signal, sampling_rate )
    print("pitch profile : {}".format("pass" if numpy.array_equal(my_pitch_profile, pitch_profile) else "FAIL"))

    data.close()

    return

def test_rqa(kind="small"):
    
    if kind=="small":
        N = 100
    elif kind=="medium":
        N = 1000
    elif kind=="large":
        N = 10000

    ###LOAD DATA FROM YYY PROTOTYPE"""
    file_name = os.path.join(os.path.dirname(__file__), 'rqa_test_'+str(N)+'.npz')

    data = numpy.load(file_name)

    x, y, m, tau, eps = data['x'], data['y'], data['m'], data['tau'], data['eps']

    Rx, SRRx, DETx, DIVx, ENTRx, LAMx = data['Rx'], data['SRRx'], data['DETx'], data['DIVx'], data['ENTRx'], data['LAMx']
    Ry, SRRy, DETy, DIVy, ENTRy, LAMy = data['Ry'], data['SRRy'], data['DETy'], data['DIVy'], data['ENTRy'], data['LAMy']
    Rc, SRRc, DETc, DIVc, ENTRc, LAMc = data['Rc'], data['SRRc'], data['DETc'], data['DIVc'], data['ENTRc'], data['LAMc']
    Rj, SRRj, DETj, DIVj, ENTRj, LAMj = data['Rj'], data['SRRj'], data['DETj'], data['DIVj'], data['ENTRj'], data['LAMj']
    
    ###CONFIRM OUTPUT MATCHES IMPLEMENTED CALPY###
    xps, yps = rqa.phase_space(x, m=m, tau=tau, eps=eps), rqa.phase_space(y, m=m, tau=tau, eps=eps)

    AA = rqa.reccurence_matrix(xps)
    BB = rqa.reccurence_matrix(yps)
    CC = rqa.reccurence_matrix(xps,yps)
    DD = rqa.reccurence_matrix(xps,yps,joint=True)

    print( AA[0] )
    print( Rx[0] )


    print(13*" "+45*"-")
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format("","x (self)","y (self)","xy cross","xy joint"))
    print(13*" "+45*"-")

    result = [
        "Rec. matrix",\
        "pass" if numpy.array_equal(AA, Rx) else "FAIL",\
        "pass" if numpy.array_equal(BB, Ry) else "FAIL",\
        "pass" if numpy.array_equal(CC, Rc) else "FAIL",\
        "pass" if numpy.array_equal(DD, Rj) else "FAIL"\
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")

    result = [
        "Rec. rate",\
        "pass" if numpy.array_equal(rqa.recurrence_rate(AA), SRRx ) else "FAIL",\
        "pass" if numpy.array_equal(rqa.recurrence_rate(BB), SRRy ) else "FAIL",\
        "pass" if numpy.array_equal(rqa.recurrence_rate(CC), SRRc ) else "FAIL",\
        "pass" if numpy.array_equal(rqa.recurrence_rate(DD), SRRj ) else "FAIL"
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")

    result = [
        "Divergence",\
        "pass" if numpy.array_equal(rqa.divergence(AA), DIVx) else "FAIL",\
        "pass" if numpy.array_equal(rqa.divergence(BB), DIVy) else "FAIL",\
        "pass" if numpy.array_equal(rqa.divergence(CC), DIVc) else "FAIL",\
        "pass" if numpy.array_equal(rqa.divergence(DD), DIVj) else "FAIL"
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")

    result = [
        "Determinism",\
        "pass" if numpy.array_equal(rqa.determinism(AA), DETx) else "FAIL",\
        "pass" if numpy.array_equal(rqa.determinism(BB), DETy) else "FAIL",\
        "pass" if numpy.array_equal(rqa.determinism(CC), DETc) else "FAIL",\
        "pass" if numpy.array_equal(rqa.determinism(DD), DETj) else "FAIL"
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")

    result = [
        "Entropy",\
        "pass" if numpy.array_equal(rqa.entropy(AA), ENTRx) else "FAIL",\
        "pass" if numpy.array_equal(rqa.entropy(BB), ENTRy) else "FAIL",\
        "pass" if numpy.array_equal(rqa.entropy(CC), ENTRc) else "FAIL",\
        "pass" if numpy.array_equal(rqa.entropy(DD), ENTRj) else "FAIL"
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")

    result = [
        "Laminarity",\
        "pass" if numpy.array_equal(rqa.laminarity(AA), LAMx) else "FAIL",\
        "pass" if numpy.array_equal(rqa.laminarity(BB), LAMy) else "FAIL",\
        "pass" if numpy.array_equal(rqa.laminarity(CC), LAMc) else "FAIL",\
        "pass" if numpy.array_equal(rqa.laminarity(DD), LAMj) else "FAIL"
    ]
    print("{: >12} | {: >8} | {: >8} | {: >8} | {: >8} |".format(*result))
    print(13*" "+45*"-")


    data.close()

    return