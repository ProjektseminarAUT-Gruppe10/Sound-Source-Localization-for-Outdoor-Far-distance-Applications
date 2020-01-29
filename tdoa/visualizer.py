import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, array, taus, valids=None, stamps=None, source_pos=None):
        self.array = array
        self.num_mics = len(array.positions)

        if stamps is None and not isinstance(taus[0][0], list):
            self.taus = [taus]
            self.stamps = [0]

            if self.valids is None:
                self.valids = None
            else:
                self.valids = [valids]
        else:
            self.taus = taus
            self.valids = valids
            self.stamps = stamps

        self.source_pos = source_pos



    def evaluate(self):
        for i in range(0, len(self.stamps)):
            fig = plt.figure()
            self.create_plot(fig, i)

        plt.show()

    def create_plot(self, fig, iteration):
        fig.suptitle("Tdoa at timestamp: " + str(self.stamps[iteration]))

        for mic in range(0, self.num_mics):
            self.create_sub_plot(fig, mic, iteration)

    
    def create_sub_plot(self, fig, mic_nr, iteration):
        #if mic_nr > 4:
        #    mic_i = mic_nr - 4 + 1
        #    upper_i = 2
        #else:
        #    mic_i = mic_nr + 1
        #    upper_i = 1
        
        sub_p = fig.add_subplot(2, 4, mic_nr + 1)

        self.add_array(sub_p)
        self.draw_connections(sub_p, mic_nr, iteration)

        if not self.source_pos is None:
            self.add_source(sub_p)
        
        sub_p.set_title("Microphone: " + str(mic_nr + 1))

    
    def add_array(self, sub_p):
        x = []
        y = []
        for pos in self.array.positions:
            x.append(pos[0])
            y.append(pos[1])

        sub_p.scatter(x, y, marker='o')


    
    def draw_connections(self, sub_p, mic_nr, iteration):
        
        for i in range(0, self.num_mics):
            if not i == mic_nr:
                tau = self.taus[iteration][mic_nr][i]
                if not self.valids is None:
                    valid = self.valids[iteration][mic_nr][i]
                else:
                    valid = True
                    
                self.draw_connection(sub_p, mic_nr, i, tau, valid)

    #https://stackoverflow.com/questions/12864294/adding-an-arbitrary-line-to-a-matplotlib-plot-in-ipython-notebook
    def draw_connection(self, sub_p, mic1, mic2, tau, valid):
        pos1 = self.array.get_position(mic1)
        pos2 = self.array.get_position(mic2)
        pos3 = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)

        if not valid:
            c = 'red'
        else:
            c = 'black'

        sub_p.annotate("",
              xy=pos1, xycoords='data',
              xytext=pos2, textcoords='data',
              arrowprops=dict(color = c,
                              arrowstyle="-",
                              connectionstyle="arc3,rad=0."), 
              )

        sub_p.annotate(str(tau),
              color = c,
              xy=pos3, xycoords='data',
              xytext=pos3, textcoords='data',
               
              )

    def add_source(self, sub_p):
        x = self.source_pos[0]
        y = self.source_pos[1]
        sub_p.scatter(x, y, marker='x')


