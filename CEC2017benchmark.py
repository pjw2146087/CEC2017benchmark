import numpy as np
import numba

from CEC2014benchmark import Bent_Cigar, Rosenbrock, Rastrigin, Expanded_Scaffer_F6, Modified_Schwefel, High_Conditioned_Elliptic, Discus, Ackley, Weierstrass, Katsuura, HappyCat, HGBat, Griewank, Expended_Griewank_plus_Rosenbrock, composition_omega, Rotated_High_Conditioned_Elliptic, Shifted_Rotated_Ackley, Rotated_Discus


@numba.njit("f8(f8[:])", nogil=True)
def Sum_of_Different_Power(x):
    dim = x.shape[0]
    x = np.abs(x)
    y = np.arange(2, dim+2, 1)
    return np.sum(np.power(x, y))


@numba.njit("f8(f8[:])", nogil=True)
def Zakharov(x):
    x = np.ascontiguousarray(x)
    t = np.sum(0.5*x)
    t *= t
    return np.dot(x, x) + t * (t + 1)


@numba.njit("f8(f8[:],f8[:])", nogil=True, fastmath=True)
def Lunacek_bi_Rastrigin(x, o):
    dim = x.shape[0]
    mu1 = -np.sqrt(5.25/(1-1/(2*np.sqrt(dim+20)-8.2)))
    x = 0.2 * np.sign(x) * (x-o) + 2.5
    z = np.diag(np.power(np.array([100] * dim), np.arange(0, dim, 1)/2/(dim-1))) * (x-2.5)

    return min(np.dot(x-2.5, x-2.5), dim+np.dot(x-mu1, x-mu1)) + 10 * (dim - np.sum(np.cos(2*np.pi*z)))


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Non_continuous_Rotated_Rastrigin(x, M, o):
    dim = x.shape[0]
    D = numba.prange(dim)
    M = np.ascontiguousarray(M)

    def return_y(_x):
        _y = np.zeros(dim)
        for i in D:
            if abs(_x[i])<=0.5:
                _y[i] = _x[i]
            else:
                _y[i] = round(2*_x[i])/2
        return _y

    def return_Tasy(_x):
        Tasy = np.zeros(dim)
        for i in D:
            if _x[i]==0:
                Tasy[i] = 0
            else:
                if _x[i]>0:
                    now_xi = np.log(_x[i])
                    Tosz = np.exp(now_xi+0.049*(np.sin(10*now_xi)+np.sin(7.9*now_xi)))
                else:
                    now_xi = np.log(abs(_x[i]))
                    Tosz = np.exp(now_xi+0.049*(np.sin(5.5*now_xi)+np.sin(3.1*now_xi)))
                if Tosz>0:
                    Tasy[i] = pow(Tosz, 1+0.2*(i-1)/(dim-1)*np.sqrt(Tosz))
                else:
                    Tasy[i] = Tosz
        return Tasy

    z = np.dot(np.dot(M, np.diag(np.power(np.array([100] * dim), np.arange(0, 30, 1)/2/(dim-1))), M), return_Tasy(return_y(np.dot(M, 0.0512*(x-o)))))
    return np.sum(np.square(z)-10*np.cos(2*np.pi*z)+10)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Levy(x):
    dim = x.shape[0]
    x = 1 + (x-1)/4
    return pow(np.sin(np.pi*x[0]), 2) + np.sum(np.square(x[0:dim-1]-1)*(1+10*np.square(np.sin(np.pi*x[0:dim-1]+1)))) + x[dim-1]*x[dim-1]*(1+pow(np.sin(2*np.pi*x[dim-1]), 2))


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Schaffer_F7(x):
    dim = x.shape[0]
    s = np.sqrt(np.square(x[0:dim-1])+np.square(x[1:dim]))
    return pow(1 / (dim-1) * np.sum(np.sqrt(s) * (np.sin(50 * np.power(s, 0.2)) + 1)), 2)


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Bent_Cigar(x, M, o):
    M = np.ascontiguousarray(M)
    return Bent_Cigar(np.dot(M, x - o)) + 100


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Sum_of_Different_Power(x, M, o):
    M = np.ascontiguousarray(M)
    return Sum_of_Different_Power(np.dot(M, x - o)) + 200


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Zakharov(x, M, o):
    M = np.ascontiguousarray(M)
    return Zakharov(np.dot(M, x - o)) + 300


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Zakharov(np.dot(M, 0.02048*(x-o)) + 1) + 400


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Rastrign(x, M, o):
    M = np.ascontiguousarray(M)
    return Rastrigin(np.dot(M, x - o)) + 500


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Schaffer_F7(x, M, o):
    M = np.ascontiguousarray(M)
    return Schaffer_F7(np.dot(M, x - o)) + 600


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Lunacek_Bi_Rastrign(x, M, o):
    M = np.ascontiguousarray(M)
    return Lunacek_bi_Rastrigin(np.dot(M, 6*(x-o)), o) + 700


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Non_Continuous_Rastrign(x, M, o):
    M = np.ascontiguousarray(M)
    return Non_continuous_Rotated_Rastrigin(np.dot(M, 0.0512*(x-o)), M, o) + 800


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Levy(x, M, o):
    M = np.ascontiguousarray(M)
    return Levy(np.dot(M, 0.0512*(x-o))) + 900


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Schwefel(x, M, o):
    M = np.ascontiguousarray(M)
    return Modified_Schwefel(np.dot(M, 10*(x - o))) + 1000


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_1(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Zakharov(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.2*dim:0.6*dim, 0.2*dim:0.6*dim]), x[0.2*dim:0.6*dim]-o[0.2*dim:0.6*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim]-o[0.6*dim:dim]))
    return  res + 1100


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_2(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[0:0.3*dim, 0:0.3*dim]), x[0:0.3*dim]-o[0:0.3*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.3*dim:0.6*dim, 0.3*dim:0.6*dim]), x[0.3*dim:0.6*dim]-o[0.3*dim:0.6*dim]))
    res += Bent_Cigar(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim]-o[0.6*dim:dim]))
    return  res + 1200


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_3(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Bent_Cigar(np.dot(np.ascontiguousarray(M[0:0.3*dim, 0:0.3*dim]), x[0:0.3*dim]-o[0:0.3*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.3*dim:0.6*dim, 0.3*dim:0.6*dim]), x[0.3*dim:0.6*dim]-o[0.3*dim:0.6*dim]))
    res += Lunacek_bi_Rastrigin(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim]-o[0.6*dim:dim]), o[0.6*dim:dim])
    return  res + 1300


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_4(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += Ackley(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Schaffer_F7(np.dot(np.ascontiguousarray(M[0.4*dim:0.6*dim, 0.4*dim:0.6*dim]), x[0.4*dim:0.6*dim]-o[0.4*dim:0.6*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.6*dim:dim, 0.6*dim:dim]), x[0.6*dim:dim]-o[0.6*dim:dim]))
    return  res + 1400


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_5(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Bent_Cigar(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += HGBat(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.4*dim:0.7*dim, 0.4*dim:0.7*dim]), x[0.4*dim:0.7*dim]-o[0.4*dim:0.7*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim]-o[0.7*dim:dim]))
    return  res + 1500


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_6(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += HGBat(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Rosenbrock(np.dot(np.ascontiguousarray(M[0.4*dim:0.7*dim, 0.4*dim:0.7*dim]), x[0.4*dim:0.7*dim]-o[0.4*dim:0.7*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim]-o[0.7*dim:dim]))
    return  res + 1600


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_7(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Katsuura(np.dot(np.ascontiguousarray(M[0:0.1*dim, 0:0.1*dim]), x[0:0.1*dim]-o[0:0.1*dim]))
    res += Ackley(np.dot(np.ascontiguousarray(M[0.1*dim:0.3*dim, 0.1*dim:0.3*dim]), x[0.1*dim:0.3*dim]-o[0.1*dim:0.3*dim]))
    res += Expended_Griewank_plus_Rosenbrock(np.dot(np.ascontiguousarray(M[0.3*dim:0.5*dim, 0.3*dim:0.5*dim]), x[0.3*dim:0.5*dim]-o[0.3*dim:0.5*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.5*dim:0.7*dim, 0.5*dim:0.7*dim]), x[0.5*dim:0.7*dim]-o[0.5*dim:0.7*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.7*dim:dim, 0.7*dim:dim]), x[0.7*dim:dim]-o[0.7*dim:dim]))
    return  res + 1700


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_8(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += Ackley(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.4*dim:0.6*dim, 0.4*dim:0.6*dim]), x[0.4*dim:0.6*dim]-o[0.4*dim:0.6*dim]))
    res += HGBat(np.dot(np.ascontiguousarray(M[0.6*dim:0.8*dim, 0.6*dim:0.8*dim]), x[0.6*dim:0.8*dim]-o[0.6*dim:0.8*dim]))
    res += Discus(np.dot(np.ascontiguousarray(M[0.8*dim:dim, 0.8*dim:dim]), x[0.8*dim:dim]-o[0.8*dim:dim]))
    return  res + 1800


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_9(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = Bent_Cigar(np.dot(np.ascontiguousarray(M[0:0.2*dim, 0:0.2*dim]), x[0:0.2*dim]-o[0:0.2*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Expended_Griewank_plus_Rosenbrock(np.dot(np.ascontiguousarray(M[0.4*dim:0.6*dim, 0.4*dim:0.6*dim]), x[0.4*dim:0.6*dim]-o[0.4*dim:0.6*dim]))
    res += Weierstrass(np.dot(np.ascontiguousarray(M[0.6*dim:0.8*dim, 0.6*dim:0.8*dim]), x[0.6*dim:0.8*dim]-o[0.6*dim:0.8*dim]))
    res += Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[0.8*dim:dim, 0.8*dim:dim]), x[0.8*dim:dim]-o[0.8*dim:dim]))
    return  res + 1900


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Hybrid_10(x, M, o):
    dim = x.shape[0]
    np.random.shuffle(x)
    res = HappyCat(np.dot(np.ascontiguousarray(M[0:0.1*dim, 0:0.1*dim]), x[0:0.1*dim]-o[0:0.1*dim]))
    res += Katsuura(np.dot(np.ascontiguousarray(M[0.1*dim:0.2*dim, 0.1*dim:0.2*dim]), x[0.1*dim:0.2*dim]-o[0.1*dim:0.2*dim]))
    res += Ackley(np.dot(np.ascontiguousarray(M[0.2*dim:0.4*dim, 0.2*dim:0.4*dim]), x[0.2*dim:0.4*dim]-o[0.2*dim:0.4*dim]))
    res += Rastrigin(np.dot(np.ascontiguousarray(M[0.4*dim:0.6*dim, 0.4*dim:0.6*dim]), x[0.4*dim:0.6*dim]-o[0.4*dim:0.6*dim]))
    res += Modified_Schwefel(np.dot(np.ascontiguousarray(M[0.6*dim:0.8*dim, 0.6*dim:0.8*dim]), x[0.6*dim:0.8*dim]-o[0.6*dim:0.8*dim]))
    res += Schaffer_F7(np.dot(np.ascontiguousarray(M[0.8*dim:dim, 0.8*dim:dim]),x[0.8*dim:dim]-o[0.8*dim:dim]))
    return  res + 2000


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_1(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30])))
    res = omega[0]*(Shifted_Rotated_Rosenbrock(x, M[0:dim,:], o[0,:])-400)
    res += omega[1]*(1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim:dim*2,:], o[1,:]) + 99.9999)
    res += omega[2]*(Shifted_Rotated_Rastrign(x, M[dim*2:dim*3,:], o[2,:])-300)
    return res + 2100


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_2(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30])))
    res = omega[0]*(Shifted_Rotated_Rastrign(x, M[0:dim,:], o[0,:])-500)
    res += omega[1]*(10*Griewank(np.dot(np.ascontiguousarray(M[dim:dim*2, :]), (x-o[1, :]))) + 100)
    res += omega[2]*(Shifted_Rotated_Schwefel(x, M[dim*2:dim*3,:], o[2,:])-800)

    return res + 2200


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_3(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40])))
    res = omega[0]*(Shifted_Rotated_Rosenbrock(x, M[0:dim,:], o[0,:])-400)
    res += omega[1]*(10*Shifted_Rotated_Ackley(x, M[dim:dim*2,:], o[1,:])-4900)
    res += omega[2]*(Shifted_Rotated_Schwefel(x, M[dim*2:dim*3,:], o[2,:])-800)
    res += omega[3]*(Shifted_Rotated_Rastrign(x, M[dim*3:dim*4,:], o[3,:])-200)

    return res + 2300


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_4(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40])))
    res = omega[0]*(10*Shifted_Rotated_Ackley(x, M[0:dim,:], o[0,:])-5000)
    res += omega[1]*(1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim:dim*2,:], o[1,:])+99.9999)
    res += omega[2]*(10*Griewank(np.dot(np.ascontiguousarray(M[dim*2:dim*3, :]), (x-o[2, :])))+200)
    res += omega[3]*(Shifted_Rotated_Rastrign(x, M[dim*3:dim*4,:], o[3,:])-200)

    return res + 2400


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_5(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50])))
    res = omega[0]*(10*Shifted_Rotated_Rastrign(x, M[0:dim,:], o[0,:])-5000)
    res += omega[1]*(HappyCat(np.dot(np.ascontiguousarray(M[dim:dim*2,:]), (x-o[1,:]))) + 100)
    res += omega[2]*(10*Shifted_Rotated_Ackley(x, M[dim*2:dim*3,:], o[2,:])-4800)
    res += omega[3]*(1e-6*Rotated_Discus(x, M[dim*3:dim*4,:], o[3,:])+296.9999)
    res += omega[4]*(Shifted_Rotated_Rosenbrock(x, M[dim*4:dim*5,:], o[4,:]))

    return res + 2500


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_6(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 20, 30, 40])))
    res = omega[0]*1e-26*Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[0:dim,:]), (x-o[0,:])))
    res += omega[1]*(10*Shifted_Rotated_Schwefel(x, M[dim:dim*2,:], o[1,:])-9900)
    res += omega[2]*(1e-6*Griewank(np.dot(np.ascontiguousarray(M[dim*2:dim*3,:]), (x-o[2,:]))) + 200)
    res += omega[3]*(10*Shifted_Rotated_Rosenbrock(x, M[dim*3:dim*4,:], o[3,:])-3700)
    res += omega[4]*(5e-4*Shifted_Rotated_Rastrign(x, M[dim*4:dim*5,:], o[4,:])-100)

    return res + 2600


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_7(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50, 60])))
    res = omega[0]*10*HGBat(np.dot(np.ascontiguousarray(M[0:dim,:]), (x-o[0,:])))
    res += omega[1]*(10*Shifted_Rotated_Rastrign(x, M[dim:dim*2,:], o[1,:])-4900)
    res += omega[2]*(2.5*Shifted_Rotated_Schwefel(x, M[dim*2:dim*3,:], o[2,:])-2300)
    res += omega[3]*(1e-26*Shifted_Rotated_Bent_Cigar(x, M[dim*3:dim*4,:], o[3,:])-1e-24 + 300)
    res += omega[4]*1e-6*(Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5,:], o[4,:])-1e-4 + 400)
    res += omega[5]*(5e-4*Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[5*dim:6*dim,:]), (x-o[5,:])))+ 500)

    return res + 2700


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_8(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50, 60])))
    res = omega[0]*(10*Shifted_Rotated_Ackley(x, M[0:dim,:], o[0,:])-5000)
    res += omega[1]*(10*Griewank(np.dot(np.ascontiguousarray(M[dim:2*dim,:]), (x-o[1,:]))) + 100)
    res += omega[2]*(1e-6*Rotated_Discus(x, M[dim*2:dim*3,:], o[2,:])-3e-4 + 200)
    res += omega[3]*(Shifted_Rotated_Rosenbrock(x, M[dim*3:dim*4,:], o[3,:])-100)
    res += omega[4]*(HappyCat(np.dot(np.ascontiguousarray(M[4*dim:5*dim,:]), (x-o[4,:]))) + 400)
    res += omega[5]*(5e-4*Expanded_Scaffer_F6(np.dot(np.ascontiguousarray(M[5*dim:6*dim,:]), (x-o[5,:]))) + 500)

    return res + 2800


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_9(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0]*(Hybrid_5(x, M[0:dim,:], o[0,:])-1500)
    res += omega[1]*(Hybrid_6(x, M[dim:dim*2,:], o[1,:])-1500)
    res += omega[2]*(Hybrid_7(x, M[dim*2:dim*3,:], o[2,:])-1500)

    return res + 2900


@numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_10(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0]*(Hybrid_5(x, M[0:dim,:], o[0,:])-1500)
    res += omega[1]*(Hybrid_8(x, M[dim:dim*2,:], o[1,:])-1700)
    res += omega[2]*(Hybrid_9(x, M[dim*2:dim*3,:], o[2,:])-1700)

    return res + 3000
