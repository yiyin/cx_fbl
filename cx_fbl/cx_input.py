


import h5py
import numpy as np

from neurokernel.LPU.InputProcessors.BaseInputProcessor import BaseInputProcessor

class BU_InputProcessor(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        uids = list(neurons.keys())
        neuron_names = [neurons[n]['name'] for n in uids]
        neuron_ids = np.array([int(name.split('/')[1][1:]) for name in neuron_names])
        neuron_side = set([name.split('/')[1][0] for name in neuron_names])
        if len(neuron_side) > 1:
            raise ValueError('BU neurons must be on one side')
        else:
            self.hemisphere = neuron_side.pop()

        self.fc = CircularGaussianFilterBank(
                        (shape[0], shape[1]),
                        rf_config.get('sigma', 0.05), 10,
                        hemisphere = self.hemisphere)

        self.index = neuron_ids - 1
        var_list = [('I', uids)]
        self.name = name
        # self.n_inputs = 80
        #self.filter_filename = '{}_filters.h5'.format(self.name)
        super(BU_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fc.create_filters()
        # self.file = h5py.File('{}_inputs.h5'.format(self.name), 'w')
        # self.file.create_dataset('I',
        #                          (0, self.n_inputs),
        #                          dtype = np.double,
        #                          maxshape=(None, self.n_inputs))

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        BU_input = (3*self.fc.apply_filters(frame)).reshape(-1)
        self.variables['I']['input'] = BU_input[self.index]
        # self.record_frame(BU_input)

    def record_frame(self, input):
        self.file['I'].resize((self.file['I'].shape[0]+1, self.n_inputs))
        self.file['I'][-1,:] = input

    def __del__(self):
        try:
            self.close_file()
        except:
            pass

class PB_InputProcessor(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        num_glomeruli = rf_config.get('num_glomeruli', 18)
        self.fr = RectangularFilterBank(shape, num_glomeruli)
        uids = list(neurons.keys())
        neuron_names = [neurons[n]['name'] for n in uids]
        neuron_side = [name.split('/')[1][0] for name in neuron_names]
        neuron_ids = np.array([(num_glomeruli//2+1 - int(name.split('/')[1][1:])) \
                     if neuron_side[i] == 'L' else\
                      int(name.split('/')[1][1:]) + num_glomeruli//2 \
                      for i, name in enumerate(neuron_names)])

        self.index = neuron_ids - 1
        var_list = [('I', uids)]
        self.name = name
        # self.n_inputs = 18
        super(PB_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fr.create_filters()
        # self.file = h5py.File('{}_inputs.h5'.format(self.name), 'w')
        # self.file.create_dataset('I',
        #                          (0, self.n_inputs),
        #                          dtype = np.double,
        #                          maxshape=(None, self.n_inputs))

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        PB_input = 3*self.fr.apply_filters(frame)

        self.variables['I']['input'] = PB_input[self.index]
        # self.record_frame(PB_input)

    def record_frame(self, input):
        self.file['I'].resize((self.file['I'].shape[0]+1, self.n_inputs))
        self.file['I'][-1,:] = input

    def __del__(self):
        try:
            self.close_file()
        except:
            pass

class CX_Video(object):
    """
    Create a test video signal.
    """
    def __init__(self, shape, dt, dur, record_file = None, record_interval = 1):
        self.shape = shape
        self.dt = dt
        self.dur = dur
        self.N_t = int(self.dur/self.dt)
        self.frame = 0
        self.record_file = record_file
        self.record_interval = record_interval
        self.record_count = 0

    def run_step(self):
        frame = self.generate_frame()
        self.frame += 1
        if self.record:
            if self.record_count == 0:
                self.record_frame()
            self.record_count = (self.record_count+1)%self.record_interval
        return frame

    def pre_run(self):
        if self.record_file is not None:
            self.file = h5py.File(self.record_file, 'w')
            self.file.create_dataset('sensory',
                                     (0, self.shape[0], self.shape[1]),
                                     dtype = np.double,
                                     maxshape=(None, self.shape[0],
                                               self.shape[1]))
            self.record = True
        else:
            self.record = False
        self.data = np.empty(self.shape, np.double)

    def record_frame(self):
        self.file['sensory'].resize((self.file['sensory'].shape[0]+1,
                                         self.shape[0], self.shape[1]))
        self.file['sensory'][-1,:,:] = self.data

    def generate_frame(self):
        pass

    def close_file(self):
        if self.record:
            self.file.close()

    def __del__(self):
        try:
            self.close_file()
        except:
            pass

class moving_bar_l2r(CX_Video):
    def __init__(self, shape, dt, dur, bar_width,
                 record_file = None, record_interval = 1):
        super(moving_bar_l2r, self).__init__(shape, dt, dur,
                                             record_file = record_file,
                                             record_interval = record_interval)
        self.bar_width = bar_width

    def generate_frame(self):
        start = int(np.ceil(self.frame*(self.shape[1]-self.bar_width)/float(self.N_t)))
        self.data.fill(0)
        self.data[:, start:start+self.bar_width] = 1.0
        return self.data

class moving_bar_r2l(CX_Video):
    def __init__(self, shape, dt, dur, bar_width,
                 record_file = None, record_interval = 1):
        super(moving_bar_r2l, self).__init__(shape, dt, dur,
                                             record_file = record_file,
                                             record_interval = record_interval)
        self.bar_width = bar_width

    def generate_frame(self):
        start = int(np.ceil(self.frame*(self.shape[1]-self.bar_width)/float(self.N_t)))
        self.data.fill(0)
        self.data[:, self.shape[1]-self.bar_width-start:-start] = 1.0
        return self.data

def Video_factory(video_class_name):
    all_video_cls = CX_Video.__subclasses__()
    all_video_names = [cls.__name__ for cls in all_video_cls]
    try:
        video_cls = all_video_cls[all_video_names.index(video_class_name)]
    except ValueError:
        print('Invalid Video subclass name: {}'.format(video_class_name))
        print('Valid names: {}'.format(all_video_names))
        return None
    return video_cls

class CircularGaussianFilterBank(object):
    """
    Create a bank of circular 2D Gaussian filters.

    Parameters
    ----------
    shape : tuple
        Image dimensions.
    sigma : float
        Parameter of Gaussian.
    n : int
        How many blocks should occupy the x-axis.
    """

    def __init__(self, shape, sigma, n, hemisphere = 'L'):
        self.shape = shape
        self.sigma = sigma
        self.n = n
        self.hemisphere = hemisphere

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_gaussian(cls, x, y, sigma):
        """
        2D Gaussian function.
        """

        return (1.0/(1*np.pi*(sigma**2)))*np.exp(-(1.0/(2*(sigma**2)))*(x**2+y**2))

    @classmethod
    def gaussian_mat(cls, shape, sigma, n_x_offset, n_y_offset, n):
        """
        Compute offset circular 2D Gaussian.
        """

        # Image dimensions in pixels:
        N_y, N_x = shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y)-(n_y_offset/float(n)/2))
        return cls.func_gaussian(X, Y, sigma)

    def create_filters(self, filename = None):
        """
        Create filter bank as order-4 tensor.
        """

        N_y, N_x = self.shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Compute how many blocks to use along the y-axis:
        n = self.n
        m = n*2*N_y//N_x

        # Construct filters offset by the blocks:
        # n_x_offsets = np.linspace(np.ceil(-n/2.0), np.floor(n/2.0), n)
        if self.hemisphere == 'L':
            n_x_offsets = np.linspace((-n+0.5)/2, -0.5/2, n)
        else:
            n_x_offsets = np.linspace(0.5/2, (n-0.5)/2, n)[::-1].copy()
        n_y_offsets = np.linspace(np.ceil(-m/2.0), np.floor(m/2.0), m)
        filters = np.empty((m, n, N_y, N_x), np.float64)
        for j, n_x_offset in enumerate(n_x_offsets):
            for i, n_y_offset in enumerate(n_y_offsets):
                filters[i, j] = self.gaussian_mat(self.shape, self.sigma,
                                                 n_x_offset, n_y_offset, n)
        self.filters = filters
        if filename is not None:
            file = h5py.File(filename,'w')
            file.create_dataset('filter', filters.shape, filters.dtype, data=filters)
            file.close()

    def apply_filters(self, frame, normalize=True):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)
        else:
            return result

class RectangularFilterBank(object):
    """
    Create a bank of 2D rectangular filters that tile the x-axis.
    """

    def __init__(self, shape, n):
        self.shape = shape
        self.n = n

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_rect(cls, x, y, width):
        return np.logical_and(x > -width/2.0, x <= width/2.0).astype(np.float64)

    @classmethod
    def rect_mat(cls, shape, n_x_offset, n):
        N_y, N_x = shape

        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y))
        return cls.func_rect(X, Y, 1.0/n)

    def create_filters(self):
        N_y, N_x = self.shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Construct filters offset by the blocks:
        n_x_offsets = np.linspace(np.ceil(-self.n/2.0), np.floor(self.n/2.0), self.n)
        filters = np.empty((self.n, N_y, N_x), np.float64)

        for j, n_x_offset in enumerate(n_x_offsets):
            filters[j] = self.rect_mat(self.shape, n_x_offset, self.n)
        self.filters = filters
        file = h5py.File('PB_filters.h5','w')
        file.create_dataset('filter', filters.shape, filters.dtype, data=filters)
        file.close()

    def apply_filters(self, frame, normalize=True):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)
        else:
            return result
