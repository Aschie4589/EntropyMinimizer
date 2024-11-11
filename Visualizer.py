import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

class SU2Visualizer:

    def __init__(self, show_sphere:bool, legend_text = ""):
        self.show_sphere = show_sphere
        self.legend_text = legend_text
        self.scatter = None
        self.legend = None
        pass

    

    def plot_stars(self,stars):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect("equal")
        if self.show_sphere:
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color="b", rstride=2, cstride = 2)
        ax.scatter(stars[0,:], stars[1,:], stars[2,:], color="r", label=self.legend_text)
        if self.legend_text:
            ax.legend(bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
        plt.show()


    def polynomial_from_vector(self, spin, vector):
        '''
        To vector 
            psi = sum_{i=-spin}^spin alpha_spin |spin>
        associate the polynomial part of the bargmann function
            psi(z) =  1/(1+|z|^2)^(n/2) * sum_{i=-spin}^{spin} \.bar{alpha_spin} sqrt{binom{2*spin}{spin-i}} * z^{spin-i}
        assume that the vector contains coeffs of spin basis in descending order: |j>, ..., |-j>
        '''
        coeffs = [np.conj(coeff)*np.sqrt(binom(int(2*spin), index)) for (index,coeff) in enumerate(vector)] 
        coeffs.reverse()        # |HW> corresponds to polynomial x^(2*HW)y^0, so we need to reverse order
        return coeffs

    def get_stars_from_polynomial(self,poly_coeffs):
        '''
        The stars for the stellar representation are obtained by performing a stereographic projection of the roots of the polynomial.
        Coefficients are assumed increasing in degree of x: [c_0, ..., c_n]

        point X+iY has inverse stereographic projection:

        2X/(1+X^2+Y^2),    2Y/(1+X^2+Y^2),      (-1+X^2+Y^2)/(1+X^2+Y^2)

        '''
        return [[2*root.real/(1+root.real**2+root.imag**2), 2*root.imag/(1+root.real**2+root.imag**2), (-1+root.real**2+root.imag**2)/(1+root.real**2+root.imag**2)] for root in np.roots(poly_coeffs)]


    def initialize_plot(self):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect("equal")

        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="b", rstride=2, cstride = 2)
        self.fig = fig
        self.ax = ax

    def update_star_plot(self, vector, J, iteration):
        stars = np.transpose(np.matrix(self.get_stars_from_polynomial(self.polynomial_from_vector(J,np.array(vector.tolist()[0])))))
        if self.scatter:
            self.scatter.remove()
        if self.legend:
            self.legend.remove()
        self.scatter = self.ax.scatter(stars[0,:], stars[1,:], stars[2,:], color="r", label=f"Iteration: {iteration}")
        self.legend = self.ax.legend(bbox_to_anchor=(1, 1), bbox_transform=self.fig.transFigure)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

