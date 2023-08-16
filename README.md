# -quantum-image-edge-detaction-
from os import lstat
from re import L
from sys import displayhook
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('bmh')
import cv2
import numpy as np
from PIL import Image
from qiskit import QuantumRegister, ClassicalRegister
from os import lstat
from re import L
from sys import displayhook
from qiskit import *
from qiskit import IBMQ
import pandas as pd
from qiskit import *
from PIL import Image
from qiskit.quantum_info import Statevector, Operator
image_size = 256 
image_crop_size = 4  
image_raw = np.array(Image.open(r"C:\Users/User\Desktop\New folder (2)\images 7.jpg"))
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(0,img.shape[1],32))
    plt.yticks(range(0,img.shape[0],32))
    plt.imshow (img,extent=[0,img.shape[1],img.shape[0],0] ,cmap='viridis')
    plt.show()
plot_image(image_raw, 'Heart Scan')
image = []
for i in range(image_size):
   image.append([])
   for j in range(image_size):
     image[i].append(image_raw[i][j][0] / 255) 
image = np.array(image)
plot_image(image, 'Heart Scan')
def amplitude_encode(img_data):  
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
    return np.array(image_norm)
j=1 
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1) 
image1=image[:32,:32].copy()
image2=image[:32,32:64].copy()
image3=image[:32,64:96].copy()
image4=image[:32,96:128].copy()
image5=image[:32,128:160].copy()
image6=image[:32,160:192].copy()
image7=image[:32,192:224].copy()
image8=image[:32,224:256].copy()
image9=image[32:64,:32].copy()
image10=image[32:64,32:64].copy()
image11=image[32:64,64:96].copy()
image12=image[32:64,96:128].copy()
image13=image[32:64,128:160].copy()
image14=image[32:64,160:192].copy()
image15=image[32:64,192:224].copy()
image16=image[32:64,224:256].copy()
image17=image[64:96,:32].copy()
image18=image[64:96,32:64].copy()
image19=image[64:96,64:96].copy()
image20=image[64:96,96:128].copy()
image21=image[64:96,128:160].copy()
image22=image[64:96,160:192].copy()
image23=image[64:96,192:224].copy()
image24=image[64:96,224:256].copy()
image25=image[96:128,:32].copy()
image26=image[96:128,32:64].copy()
image27=image[96:128,64:96].copy()
image28=image[96:128,96:128].copy()
image29=image[96:128,128:160].copy()
image30=image[96:128,160:192].copy()
image31=image[96:128,192:224].copy()
image32=image[96:128,224:256].copy()
image33=image[128:160,:32].copy()
image34=image[128:160,32:64].copy()
image35=image[128:160,64:96].copy()
image36=image[128:160,96:128].copy()
image37=image[128:160,128:160].copy()
image38=image[128:160,160:192].copy()
image39=image[128:160,192:224].copy()
image40=image[128:160,224:256].copy()
image41=image[160:192,:32].copy()
image42=image[160:192,32:64].copy()
image43=image[160:192,64:96].copy()
image44=image[160:192,96:128].copy()
image45=image[160:192,128:160].copy()
image46=image[160:192,160:192].copy()
image47=image[160:192,192:224].copy()
image48=image[160:192,224:256].copy()
image49=image[192:224,:32].copy()
image50=image[192:224,32:64].copy()
image51=image[192:224,64:96].copy()
image52=image[192:224,96:128].copy()
image53=image[192:224,128:160].copy()
image54=image[192:224,160:192].copy()
image55=image[192:224,192:224].copy()
image56=image[192:224,224:256].copy()
image57=image[224:256,:32].copy()
image58=image[224:256,32:64].copy()
image59=image[224:256,64:96].copy()
image60=image[224:256,96:128].copy()
image61=image[224:256,128:160].copy()
image62=image[224:256,160:192].copy()
image63=image[224:256,192:224].copy()
image64=image[224:256,224:256].copy()
image_norm_h = amplitude_encode(image1)
image_norm_v = amplitude_encode(image1.T)
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
qc_h = QuantumCircuit(total_qb)
qc_h.initialize(image_norm_h, range(1, total_qb))
qc_h.h(0)
qc_h.unitary(D2n_1, range(total_qb))
qc_h.h(0)
qc_v = QuantumCircuit(total_qb)
qc_v.initialize(image_norm_v, range(1, total_qb))
qc_v.h(0)
qc_v.unitary(D2n_1, range(total_qb))
qc_v.h(0)
circ_list = [qc_h, qc_v]
back = Aer.get_backend('statevector_simulator')
results = execute(circ_list, backend=back).result()
sv_h = results.get_statevector(qc_h)
sv_v = results.get_statevector(qc_v)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan_h1=np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan_h2=np.abs(np.array([1 if threshold1(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan_h3=np.roll(edge_scan_h2,1)
edge_scan_h4=edge_scan_h3.reshape(32,32)
edge_scan_h=edge_scan_h1|edge_scan_h4
edge_scan_v1 = np.abs(np.array([1 if threshold(sv_v[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan_v2=np.abs(np.array([1 if threshold1(sv_v[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan_v3=np.roll(edge_scan_v2,1)
edge_scan_v4=edge_scan_v3.reshape(32,32)
edge_scan_v=edge_scan_v1|edge_scan_v4
edge_scan_sim = edge_scan_h | edge_scan_v
x=[]
x.append(edge_scan_sim)
for i in [image2,image3,image4,image5,image6,image7,image8]:
    image_norm_h = amplitude_encode(i)
    image_norm_v = amplitude_encode(i.T)
    data_qb =10
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    circ_list = [qc_h, qc_v]
    back = Aer.get_backend('statevector_simulator')
    results = execute(circ_list, backend=back).result()
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan_h1=np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan_h2=np.abs(np.array([1 if threshold1(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan_h3=np.roll(edge_scan_h2,1)
    edge_scan_h4=edge_scan_h3.reshape(32,32)
    edge_scan_h=edge_scan_h1|edge_scan_h4
    edge_scan_v1 = np.abs(np.array([1 if threshold(sv_v[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan_v2=np.abs(np.array([1 if threshold1(sv_v[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan_v3=np.roll(edge_scan_v2,1)
    edge_scan_v4=edge_scan_v3.reshape(32,32)
    edge_scan_v=edge_scan_v1|edge_scan_v4
    edge_scan_sim = edge_scan_h | edge_scan_v
    x.append(edge_scan_sim)
    image_=np.concatenate((x[j-1],x[j]),axis=1)
    x.append(image_)
    j=j+2
y=[]
k=1
image_norm_h9 = amplitude_encode(image9)
image_norm_v9 = amplitude_encode(image9.T)
qc_h9 = QuantumCircuit(total_qb)
qc_h9.initialize(image_norm_h9, range(1, total_qb))
qc_h9.h(0)
qc_h9.unitary(D2n_1, range(total_qb))
qc_h9.h(0)
qc_v9 = QuantumCircuit(total_qb)
qc_v9.initialize(image_norm_v9, range(1, total_qb))
qc_v9.h(0)
qc_v9.unitary(D2n_1, range(total_qb))
qc_v9.h(0)
circ_list9 = [qc_h9, qc_v9]
back = Aer.get_backend('statevector_simulator')
results = execute(circ_list9, backend=back).result()
sv_h9 = results.get_statevector(qc_h9)
sv_v9 = results.get_statevector(qc_v9)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan9_h1=np.abs(np.array([1 if threshold(sv_h9[2*k+1].real) else 0 for k in range (2**data_qb) ])).reshape(32,32)
edge_scan9_h2=np.abs(np.array([1 if threshold1(sv_h9[2*k+1].real) else 0 for k in range (2**data_qb) ]))
edge_scan9_h3=np.roll(edge_scan9_h2,1)
edge_scan9_h4=edge_scan9_h3.reshape(32,32)
edge_scan9_h=edge_scan9_h1|edge_scan9_h4
edge_scan9_v1 = np.abs(np.array([1 if threshold(sv_v9[2*k+1].real)  else 0 for k in range(2**data_qb)])).reshape(32,32).T
edge_scan9_v2=np.abs(np.array([1 if threshold1(sv_v9[2*k+1].real) else 0 for k in range (2**data_qb) ]))
edge_scan9_v3  =np.roll(edge_scan9_v2,1)
edge_scan9_v4=edge_scan9_v3.reshape(32,32)
edge_scan9_v=edge_scan9_v1|edge_scan9_v4
edge_scan_sim9=edge_scan9_h | edge_scan9_v
y.append(edge_scan_sim9)
for i in [image10,image11,image12,image13,image14,image15,image16]  : 
    image_norm_h9 = amplitude_encode(i)
    image_norm_v9 = amplitude_encode(i.T)
    qc_h9 = QuantumCircuit(total_qb)
    qc_h9.initialize(image_norm_h9, range(1, total_qb))
    qc_h9.h(0)
    qc_h9.unitary(D2n_1, range(total_qb))
    qc_h9.h(0)
    qc_v9 = QuantumCircuit(total_qb)
    qc_v9.initialize(image_norm_v9, range(1, total_qb))
    qc_v9.h(0)
    qc_v9.unitary(D2n_1, range(total_qb))
    qc_v9.h(0)
    circ_list9 = [qc_h9, qc_v9]
    back = Aer.get_backend('statevector_simulator')
    results = execute(circ_list9, backend=back).result()
    sv_h9 = results.get_statevector(qc_h9)
    sv_v9 = results.get_statevector(qc_v9)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan9_h1=np.abs(np.array([1 if threshold(sv_h9[2*k+1].real) else 0 for k in range (2**data_qb) ])).reshape(32,32)
    edge_scan9_h2=np.abs(np.array([1 if threshold1(sv_h9[2*k+1].real) else 0 for k in range (2**data_qb) ]))
    edge_scan9_h3=np.roll(edge_scan9_h2,1)
    edge_scan9_h4=edge_scan9_h3.reshape(32,32)
    edge_scan9_h=edge_scan9_h1|edge_scan9_h4
    edge_scan9_v1 = np.abs(np.array([1 if threshold(sv_v9[2*k+1].real)  else 0 for k in range(2**data_qb)])).reshape(32,32).T
    edge_scan9_v2=np.abs(np.array([1 if threshold1(sv_v9[2*k+1].real) else 0 for k in range (2**data_qb) ]))
    edge_scan9_v3=np.roll(edge_scan9_v2,1)
    edge_scan9_v4=edge_scan9_v3.reshape(32,32)
    edge_scan9_v=edge_scan9_v1|edge_scan9_v4
    edge_scan_sim9=edge_scan9_h | edge_scan9_v
    y.append(edge_scan_sim9)
    image_9=np.concatenate((y[k-1],y[k]),axis=1)
    y.append(image_9)
    k=k+2
image_ultimate1=np.concatenate((x[14],y[14]),axis=0)
image_norm_h3 = amplitude_encode(image17)
image_norm_v3 = amplitude_encode(image17.T)
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
qc_h3 = QuantumCircuit(total_qb)
qc_h3.initialize(image_norm_h3, range(1, total_qb))
qc_h3.h(0)
qc_h3.unitary(D2n_1, range(total_qb))
qc_h3.h(0)
qc_v3 = QuantumCircuit(total_qb)
qc_v3.initialize(image_norm_v3, range(1, total_qb))
qc_v3.h(0)
qc_v3.unitary(D2n_1, range(total_qb))
qc_v3.h(0)
circ_list3 = [qc_h3, qc_v3]
back = Aer.get_backend('statevector_simulator')
results3 = execute(circ_list3, backend=back).result()
sv_h3 = results3.get_statevector(qc_h3)
sv_v3 = results3.get_statevector(qc_v3)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan3_h1=np.abs(np.array([1 if threshold(sv_h3[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan3_h2=np.abs(np.array([1 if threshold1(sv_h3[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan3_h3=np.roll(edge_scan3_h2,1)
edge_scan3_h4=edge_scan3_h3.reshape(32,32)
edge_scan3_h=edge_scan3_h1|edge_scan3_h4
edge_scan3_v1 = np.abs(np.array([1 if threshold(sv_v3[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan3_v2=np.abs(np.array([1 if threshold1(sv_v3[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan3_v3=np.roll(edge_scan3_v2,1)
edge_scan3_v4=edge_scan3_v3.reshape(32,32)
edge_scan3_v=edge_scan3_v1|edge_scan3_v4
edge_scan_sim3 = edge_scan3_h | edge_scan3_v
z=[]
l=1
z.append(edge_scan_sim3)
for i in [image18,image19,image20,image21,image22,image23,image24] :
    image_norm_h3 = amplitude_encode(i)
    image_norm_v3 = amplitude_encode(i.T)
    qc_h3 = QuantumCircuit(total_qb)
    qc_h3.initialize(image_norm_h3, range(1, total_qb))
    qc_h3.h(0)
    qc_h3.unitary(D2n_1, range(total_qb))
    qc_h3.h(0)
    qc_v3 = QuantumCircuit(total_qb)
    qc_v3.initialize(image_norm_v3, range(1, total_qb))
    qc_v3.h(0)
    qc_v3.unitary(D2n_1, range(total_qb))
    qc_v3.h(0)
    circ_list3 = [qc_h3, qc_v3]
    back = Aer.get_backend('statevector_simulator')
    results3 = execute(circ_list3, backend=back).result()
    sv_h3 = results3.get_statevector(qc_h3)
    sv_v3 = results3.get_statevector(qc_v3)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan3_h1=np.abs(np.array([1 if threshold(sv_h3[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan3_h2=np.abs(np.array([1 if threshold1(sv_h3[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan3_h3=np.roll(edge_scan3_h2,1)
    edge_scan3_h4=edge_scan3_h3.reshape(32,32)
    edge_scan3_h=edge_scan3_h1|edge_scan3_h4
    edge_scan3_v1 = np.abs(np.array([1 if threshold(sv_v3[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan3_v2=np.abs(np.array([1 if threshold1(sv_v3[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan3_v3=np.roll(edge_scan3_v2,1)
    edge_scan3_v4=edge_scan3_v3.reshape(32,32)
    edge_scan3_v=edge_scan3_v1|edge_scan3_v4
    edge_scan_sim3 = edge_scan3_h | edge_scan3_v
    z.append(edge_scan_sim3)
    edge_emage3=np.concatenate((z[l-1],z[l]),axis=1)
    z.append(edge_emage3) 
    l=l+2
image_ultimate2=np.concatenate((image_ultimate1,z[14]),axis=0)  
image_norm_h4 = amplitude_encode(image25)
image_norm_v4 = amplitude_encode(image25.T)
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
qc_h4 = QuantumCircuit(total_qb)
qc_h4.initialize(image_norm_h4, range(1, total_qb))
qc_h4.h(0)
qc_h4.unitary(D2n_1, range(total_qb))
qc_h4.h(0)
qc_v4 = QuantumCircuit(total_qb)
qc_v4.initialize(image_norm_v4, range(1, total_qb))
qc_v4.h(0)
qc_v4.unitary(D2n_1, range(total_qb))
qc_v4.h(0)
circ_list4 = [qc_h4, qc_v4]
back = Aer.get_backend('statevector_simulator')
results4 = execute(circ_list4, backend=back).result()
sv_h4 = results4.get_statevector(qc_h4)
sv_v4 = results4.get_statevector(qc_v4)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan4_h1=np.abs(np.array([1 if threshold(sv_h4[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan4_h2=np.abs(np.array([1 if threshold1(sv_h4[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan4_h3=np.roll(edge_scan4_h2,1)
edge_scan4_h4=edge_scan4_h3.reshape(32,32)
edge_scan4_h=edge_scan4_h1|edge_scan4_h4
edge_scan4_v1 = np.abs(np.array([1 if threshold(sv_v4[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan4_v2=np.abs(np.array([1 if threshold1(sv_v4[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan4_v3=np.roll(edge_scan4_v2,1)
edge_scan4_v4=edge_scan4_v3.reshape(32,32)
edge_scan4_v=edge_scan4_v1|edge_scan4_v4
edge_scan_sim4 = edge_scan4_h | edge_scan4_v
a=[]
a.append(edge_scan_sim4)
m=1
for i in [image26,image27,image28,image29,image30,image31,image32] :
    image_norm_h4 = amplitude_encode(i)
    image_norm_v4 = amplitude_encode(i.T)
    qc_h4 = QuantumCircuit(total_qb)
    qc_h4.initialize(image_norm_h4, range(1, total_qb))
    qc_h4.h(0)
    qc_h4.unitary(D2n_1, range(total_qb))
    qc_h4.h(0)
    qc_v4 = QuantumCircuit(total_qb)
    qc_v4.initialize(image_norm_v4, range(1, total_qb))
    qc_v4.h(0)
    qc_v4.unitary(D2n_1, range(total_qb))
    qc_v4.h(0)
    circ_list4 = [qc_h4, qc_v4]
    back = Aer.get_backend('statevector_simulator')
    results4 = execute(circ_list4, backend=back).result()
    sv_h4 = results4.get_statevector(qc_h4)
    sv_v4 = results4.get_statevector(qc_v4)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan4_h1=np.abs(np.array([1 if threshold(sv_h4[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan4_h2=np.abs(np.array([1 if threshold1(sv_h4[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan4_h3=np.roll(edge_scan4_h2,1)
    edge_scan4_h4=edge_scan4_h3.reshape(32,32)
    edge_scan4_h=edge_scan4_h1|edge_scan4_h4
    edge_scan4_v1 = np.abs(np.array([1 if threshold(sv_v4[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan4_v2=np.abs(np.array([1 if threshold1(sv_v4[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan4_v3=np.roll(edge_scan4_v2,1)
    edge_scan4_v4=edge_scan4_v3.reshape(32,32)
    edge_scan4_v=edge_scan4_v1|edge_scan4_v4
    edge_scan_sim4 = edge_scan4_h | edge_scan4_v
    a.append(edge_scan_sim4)
    edge_emage4=np.concatenate((a[m-1],a[m]),axis=1)
    a.append(edge_emage4)
    m=m+2
image_ultimate3=np.concatenate((image_ultimate2,a[14]),axis=0) 
image_norm_h5 = amplitude_encode(image33)
image_norm_v5 = amplitude_encode(image33.T)
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
qc_h5 = QuantumCircuit(total_qb)
qc_h5.initialize(image_norm_h5, range(1, total_qb))
qc_h5.h(0)
qc_h5.unitary(D2n_1, range(total_qb))
qc_h5.h(0)
qc_v5 = QuantumCircuit(total_qb)
qc_v5.initialize(image_norm_v5, range(1, total_qb))
qc_v5.h(0)
qc_v5.unitary(D2n_1, range(total_qb))
qc_v5.h(0)
circ_list5 = [qc_h5, qc_v5]
back = Aer.get_backend('statevector_simulator')
results5 = execute(circ_list5, backend=back).result()
sv_h5 = results5.get_statevector(qc_h5)
sv_v5 = results5.get_statevector(qc_v5)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan5_h1=np.abs(np.array([1 if threshold(sv_h5[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan5_h2=np.abs(np.array([1 if threshold1(sv_h5[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan5_h3=np.roll(edge_scan5_h2,1)
edge_scan5_h4=edge_scan5_h3.reshape(32,32)
edge_scan5_h=edge_scan5_h1|edge_scan5_h4
edge_scan5_v1 = np.abs(np.array([1 if threshold(sv_v5[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan5_v2=np.abs(np.array([1 if threshold1(sv_v5[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan5_v3=np.roll(edge_scan5_v2,1)
edge_scan5_v4=edge_scan5_v3.reshape(32,32)
edge_scan5_v=edge_scan5_v1|edge_scan5_v4
edge_scan_sim5 = edge_scan5_h | edge_scan5_v
b=[]
b.append(edge_scan_sim5)
n=1
for i in [image34,image35,image36,image37,image38,image39,image40] :
    image_norm_h5 = amplitude_encode(i)
    image_norm_v5 = amplitude_encode(i.T)
    qc_h5 = QuantumCircuit(total_qb)
    qc_h5.initialize(image_norm_h5, range(1, total_qb))
    qc_h5.h(0)
    qc_h5.unitary(D2n_1, range(total_qb))
    qc_h5.h(0)
    qc_v5 = QuantumCircuit(total_qb)
    qc_v5.initialize(image_norm_v5, range(1, total_qb))
    qc_v5.h(0)
    qc_v5.unitary(D2n_1, range(total_qb))
    qc_v5.h(0)
    circ_list5 = [qc_h5, qc_v5]
    back = Aer.get_backend('statevector_simulator')
    results5 = execute(circ_list5, backend=back).result()
    sv_h5 = results5.get_statevector(qc_h5)
    sv_v5 = results5.get_statevector(qc_v5)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan5_h1=np.abs(np.array([1 if threshold(sv_h5[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan5_h2=np.abs(np.array([1 if threshold1(sv_h5[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan5_h3=np.roll(edge_scan5_h2,1)
    edge_scan5_h4=edge_scan5_h3.reshape(32,32)
    edge_scan5_h=edge_scan5_h1|edge_scan5_h4
    edge_scan5_v1 = np.abs(np.array([1 if threshold(sv_v5[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan5_v2=np.abs(np.array([1 if threshold1(sv_v5[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan5_v3=np.roll(edge_scan5_v2,1)
    edge_scan5_v4=edge_scan5_v3.reshape(32,32)
    edge_scan5_v=edge_scan5_v1|edge_scan5_v4
    edge_scan_sim5 = edge_scan5_h | edge_scan5_v
    b.append(edge_scan_sim5)
    edge_emage5=np.concatenate((b[n-1],b[n]),axis=1)
    b.append(edge_emage5)
    n=n+2
image_ultimate4=np.concatenate((image_ultimate3,b[14]),axis=0)
image_norm_h6 = amplitude_encode(image41)
image_norm_v6 = amplitude_encode(image41.T)
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
qc_h6 = QuantumCircuit(total_qb)
qc_h6.initialize(image_norm_h6, range(1, total_qb))
qc_h6.h(0)
qc_h6.unitary(D2n_1, range(total_qb))
qc_h6.h(0)
qc_v6 = QuantumCircuit(total_qb)
qc_v6.initialize(image_norm_v6, range(1, total_qb))
qc_v6.h(0)
qc_v6.unitary(D2n_1, range(total_qb))
qc_v6.h(0)
circ_list6 = [qc_h6, qc_v6]
back = Aer.get_backend('statevector_simulator')
results6 = execute(circ_list6, backend=back).result()
sv_h6 = results6.get_statevector(qc_h6)
sv_v6 = results6.get_statevector(qc_v6)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan6_h1=np.abs(np.array([1 if threshold(sv_h6[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan6_h2=np.abs(np.array([1 if threshold1(sv_h6[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan6_h3=np.roll(edge_scan6_h2,1)
edge_scan6_h4=edge_scan6_h3.reshape(32,32)
edge_scan6_h=edge_scan6_h1|edge_scan6_h4
edge_scan6_v1 = np.abs(np.array([1 if threshold(sv_v6[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan6_v2=np.abs(np.array([1 if threshold1(sv_v6[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan6_v3=np.roll(edge_scan6_v2,1)
edge_scan6_v4=edge_scan6_v3.reshape(32,32)
edge_scan6_v=edge_scan6_v1|edge_scan6_v4
edge_scan_sim6 = edge_scan6_h | edge_scan6_v
c=[]
c.append(edge_scan_sim6)
o=1
for i in [image42,image43,image44,image45,image46,image47,image48] :
    image_norm_h6 = amplitude_encode(i)
    image_norm_v6 = amplitude_encode(i.T)
    qc_h6 = QuantumCircuit(total_qb)
    qc_h6.initialize(image_norm_h6, range(1, total_qb))
    qc_h6.h(0)
    qc_h6.unitary(D2n_1, range(total_qb))
    qc_h6.h(0)
    qc_v6 = QuantumCircuit(total_qb)
    qc_v6.initialize(image_norm_v6, range(1, total_qb))
    qc_v6.h(0)
    qc_v6.unitary(D2n_1, range(total_qb))
    qc_v6.h(0)
    circ_list6 = [qc_h6, qc_v6]
    back = Aer.get_backend('statevector_simulator')
    results6 = execute(circ_list6, backend=back).result()
    sv_h6 = results6.get_statevector(qc_h6)
    sv_v6 = results6.get_statevector(qc_v6)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan6_h1=np.abs(np.array([1 if threshold(sv_h6[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan6_h2=np.abs(np.array([1 if threshold1(sv_h6[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan6_h3=np.roll(edge_scan6_h2,1)
    edge_scan6_h4=edge_scan6_h3.reshape(32,32)
    edge_scan6_h=edge_scan6_h1|edge_scan6_h4
    edge_scan6_v1 = np.abs(np.array([1 if threshold(sv_v6[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan6_v2=np.abs(np.array([1 if threshold1(sv_v6[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan6_v3=np.roll(edge_scan6_v2,1)
    edge_scan6_v4=edge_scan6_v3.reshape(32,32)
    edge_scan6_v=edge_scan6_v1|edge_scan6_v4
    edge_scan_sim6 = edge_scan6_h | edge_scan6_v
    c.append(edge_scan_sim6)
    edge_emage6=np.concatenate((c[o-1],c[o]),axis=1)
    c.append(edge_emage6)
    o=o+2
image_ultimate5=np.concatenate((image_ultimate4,c[14]),axis=0)
image_norm_h7 = amplitude_encode(image49)
image_norm_v7 = amplitude_encode(image49.T)
qc_h7 = QuantumCircuit(total_qb)
qc_h7.initialize(image_norm_h7, range(1, total_qb))
qc_h7.h(0)
qc_h7.unitary(D2n_1, range(total_qb))
qc_h7.h(0)
qc_v7 = QuantumCircuit(total_qb)
qc_v7.initialize(image_norm_v7, range(1, total_qb))
qc_v7.h(0)
qc_v7.unitary(D2n_1, range(total_qb))
qc_v7.h(0)
circ_list7 = [qc_h7, qc_v7]
back = Aer.get_backend('statevector_simulator')
results7 = execute(circ_list7, backend=back).result()
sv_h7 = results7.get_statevector(qc_h7)
sv_v7 = results7.get_statevector(qc_v7)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan7_h1=np.abs(np.array([1 if threshold(sv_h7[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan7_h2=np.abs(np.array([1 if threshold1(sv_h7[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan7_h3=np.roll(edge_scan7_h2,1)
edge_scan7_h4=edge_scan7_h3.reshape(32,32)
edge_scan7_h=edge_scan7_h1|edge_scan7_h4
edge_scan7_v1 = np.abs(np.array([1 if threshold(sv_v7[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan7_v2=np.abs(np.array([1 if threshold1(sv_v7[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan7_v3=np.roll(edge_scan7_v2,1)
edge_scan7_v4=edge_scan7_v3.reshape(32,32)
edge_scan7_v=edge_scan7_v1|edge_scan7_v4
edge_scan_sim7 = edge_scan7_h | edge_scan7_v
d=[]
d.append(edge_scan_sim7)
p=1
for i in [image50,image51,image52,image53,image54,image55,image56]:
    image_norm_h7 = amplitude_encode(i)
    image_norm_v7 = amplitude_encode(i.T)
    qc_h7 = QuantumCircuit(total_qb)
    qc_h7.initialize(image_norm_h7, range(1, total_qb))
    qc_h7.h(0)
    qc_h7.unitary(D2n_1, range(total_qb))
    qc_h7.h(0)
    qc_v7 = QuantumCircuit(total_qb)
    qc_v7.initialize(image_norm_v7, range(1, total_qb))
    qc_v7.h(0)
    qc_v7.unitary(D2n_1, range(total_qb))
    qc_v7.h(0)
    circ_list7 = [qc_h7, qc_v7]
    back = Aer.get_backend('statevector_simulator')
    results7 = execute(circ_list7, backend=back).result()
    sv_h7 = results7.get_statevector(qc_h7)
    sv_v7 = results7.get_statevector(qc_v7)
    threshold = lambda amp: ( amp <-1e-15)
    threshold1 = lambda amp1 : (amp1>1e-2)
    edge_scan7_h1=np.abs(np.array([1 if threshold(sv_h7[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
    edge_scan7_h2=np.abs(np.array([1 if threshold1(sv_h7[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan7_h3=np.roll(edge_scan7_h2,1)
    edge_scan7_h4=edge_scan7_h3.reshape(32,32)
    edge_scan7_h=edge_scan7_h1|edge_scan7_h4
    edge_scan7_v1 = np.abs(np.array([1 if threshold(sv_v7[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
    edge_scan7_v2=np.abs(np.array([1 if threshold1(sv_v7[2*i+1].real) else 0 for i in range (2**data_qb) ]))
    edge_scan7_v3=np.roll(edge_scan7_v2,1)
    edge_scan7_v4=edge_scan7_v3.reshape(32,32)
    edge_scan7_v=edge_scan7_v1|edge_scan7_v4
    edge_scan_sim7 = edge_scan7_h | edge_scan7_v
    d.append(edge_scan_sim7)
    edge_emage7=np.concatenate((d[p-1],d[p]),axis=1)
    d.append(edge_emage7)
    p=p+2
image_ultimate6=np.concatenate((image_ultimate5,d[14]),axis=0)
image_norm_h8 = amplitude_encode(image57)
image_norm_v8 = amplitude_encode(image57.T)
qc_h8 = QuantumCircuit(total_qb)
qc_h8.initialize(image_norm_h8, range(1, total_qb))
qc_h8.h(0)
qc_h8.unitary(D2n_1, range(total_qb))
qc_h8.h(0)
qc_v8 = QuantumCircuit(total_qb)
qc_v8.initialize(image_norm_v8, range(1, total_qb))
qc_v8.h(0)
qc_v8.unitary(D2n_1, range(total_qb))
qc_v8.h(0)
circ_list8 = [qc_h8, qc_v8]
back = Aer.get_backend('statevector_simulator')
results8 = execute(circ_list8, backend=back).result()
sv_h8 = results8.get_statevector(qc_h8)
sv_v8 = results8.get_statevector(qc_v8)
threshold = lambda amp: ( amp <-1e-15)
threshold1 = lambda amp1 : (amp1>1e-2)
edge_scan8_h1=np.abs(np.array([1 if threshold(sv_h8[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
edge_scan8_h2=np.abs(np.array([1 if threshold1(sv_h8[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan8_h3=np.roll(edge_scan8_h2,1)
edge_scan8_h4=edge_scan8_h3.reshape(32,32)
edge_scan8_h=edge_scan8_h1|edge_scan8_h4
edge_scan8_v1 = np.abs(np.array([1 if threshold(sv_v8[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
edge_scan8_v2=np.abs(np.array([1 if threshold1(sv_v8[2*i+1].real) else 0 for i in range (2**data_qb) ]))
edge_scan8_v3=np.roll(edge_scan8_v2,1)
edge_scan8_v4=edge_scan8_v3.reshape(32,32)
edge_scan8_v=edge_scan8_v1|edge_scan8_v4
edge_scan_sim8 = edge_scan8_h | edge_scan8_v
e=[]
e.append(edge_scan_sim8)
q=1
for i in [image58,image59,image60,image61,image62,image63,image64]:
   image_norm_h8 = amplitude_encode(i)
   image_norm_v8 = amplitude_encode(i.T)
   data_qb = 10
   anc_qb = 1
   total_qb = data_qb + anc_qb
   D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
   qc_h8 = QuantumCircuit(total_qb)
   qc_h8.initialize(image_norm_h8, range(1, total_qb))
   qc_h8.h(0)
   qc_h8.unitary(D2n_1, range(total_qb))
   qc_h8.h(0)
   qc_v8 = QuantumCircuit(total_qb)
   qc_v8.initialize(image_norm_v8, range(1, total_qb))
   qc_v8.h(0)
   qc_v8.unitary(D2n_1, range(total_qb))
   qc_v8.h(0)
   circ_list8 = [qc_h8, qc_v8]
   back = Aer.get_backend('statevector_simulator')
   results8 = execute(circ_list8, backend=back).result()
   sv_h8 = results8.get_statevector(qc_h8)
   sv_v8 = results8.get_statevector(qc_v8)
   threshold = lambda amp: ( amp <-1e-15)
   threshold1 = lambda amp1 : (amp1>1e-2)
   edge_scan8_h1=np.abs(np.array([1 if threshold(sv_h8[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(32,32)
   edge_scan8_h2=np.abs(np.array([1 if threshold1(sv_h8[2*i+1].real) else 0 for i in range (2**data_qb) ]))
   edge_scan8_h3=np.roll(edge_scan8_h2,1)
   edge_scan8_h4=edge_scan8_h3.reshape(32,32)
   edge_scan8_h=edge_scan8_h1|edge_scan8_h4
   edge_scan8_v1 = np.abs(np.array([1 if threshold(sv_v8[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(32,32).T
   edge_scan8_v2=np.abs(np.array([1 if threshold1(sv_v8[2*i+1].real) else 0 for i in range (2**data_qb) ]))
   edge_scan8_v3=np.roll(edge_scan8_v2,1)
   edge_scan8_v4=edge_scan8_v3.reshape(32,32)
   edge_scan8_v=edge_scan8_v1|edge_scan8_v4
   edge_scan_sim8 = edge_scan8_h | edge_scan8_v
   e.append(edge_scan_sim8)
   edge_emage8=np.concatenate((e[q-1],e[q]),axis=1)
   e.append(edge_emage8)
   q=q+2
image_ultimate7=np.concatenate((image_ultimate6,e[14]),axis=0)
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(0,img.shape[1],32))
    plt.yticks(range(0,img.shape[0],32))
    plt.imshow (img,extent=[0,img.shape[1],img.shape[0],0] ,cmap='viridis')
    plt.show()
plot_image(image_ultimate7, 'Edge Detected image')
