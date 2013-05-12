import pylab
import sys

####################################################################
######################## PLOT INTERACTIVITY ########################
####################################################################

class MouseMonitor:

    def __init__(self):
        self.event = None
        self.xdatalist = []
        self.ymax = []
        self.ydatalist = []
        self.length = []

    def my_call(self, event):
        self.event = event
       
        if event.button == 1: ## Button #1 is the left click ## 
            self.xdatalist.append(event.xdata)
            self.ydatalist.append(event.ydata)
            
            sys.stdout.flush()
            print 'x = %s and y = %s' % (event.xdata,event.ydata)
            sys.stdout.flush()
            ax = pylab.gca()
            ax.hold(True)
            ax.plot([event.xdata],[event.ydata],'r+',markersize=9)
            pylab.draw()

        if event.button == 3: ## Button #3 is the right click ## 
            self.length.append(len(self.xdatalist))
            pylab.disconnect(self.cid)
            self.ask_for_another()
        
        if event.button == 2: ## Button #3 is the right click ## 
            self.ymax.append(event.ydata)
            sys.stdout.flush()
            ax = pylab.gca()
            ax.hold(True)
            ax.plot([event.xdata],[event.ydata],'go')
            pylab.draw()

            
    def ask_for_another(self): ## The plot window closes ## 
        sys.stdout.flush()
        self.plot_next()

#    def return_data(self):
#        return self.xdatalist, self.ydatalist

    def return_length(self):
        return self.length

    def set_data(self,x,y):
        self.xlist = x
        self.ylist = y
        self.i = 0

    def set_data2(self,x1,y1):
        self.xlist1 = x1
        self.ylist1 = y1
        self.i = 0

    def plot_next(self):
        if self.i < len(self.xlist):
            pylab.cla()
            pylab.plot(self.xlist[self.i],self.ylist[self.i])
            pylab.ylim(min(self.ylist[self.i]),max(self.ylist[self.i]))
            pylab.xlim(min(self.xlist[self.i]),max(self.xlist[self.i]))
            pylab.axhline(0.0)
            self.i += 1
            self.connect_to_plot()
            pylab.draw()
        else:
            sys.stdout.flush()
            print 'all out of data.'

    def connect_to_plot(self):
        self.cid = pylab.connect('button_press_event',self.my_call)

