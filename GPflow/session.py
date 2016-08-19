import tensorflow as tf
import abc
import os
from tensorflow.python.client import timeline

class TracerSessionAbstract(tf.Session):
	#This pattern is a little odd.
	#TracerSession is an abstract base class.
	#You need to inherit from it and make a function
	#that implements getOutputDict.
	__metaclass__ = abc.ABCMeta	

	def __init__(self, *args, **kwargs):
		super(TracerSessionAbstract, self).__init__(*args, **kwargs)
		self.counter=0
		self.profiler_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		self.outputFileName,self.outputDirectory,self.eachTime = self.getOutputParams()
		if self.outputDirectory!=None:
			if os.path.isfile(self.outputDirectory):
				raise IOError("In tracer: given directory name is a file.")
			if not(os.path.isdir(self.outputDirectory)):
				os.path.mkdir(self.outputDirectory)
	
	# getOutputParams needs to return 
	# outputFileName, outputDirectory, eachTime
	@abc.abstractmethod
	def getOutputParams(self):
		return
    
	def getFileName(self):
		if self.outputDirectory!=None:
			dirStub = self.outputDirectory
		else:
			dirStub = ''
		if self.eachTime:
			return os.path.join(dirStub,self.outputFileName+'_'+str(counter)+'.json')
		else:
			return os.path.join(dirStub,self.outputFileName+'.json')
        
        
	def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
		#Make sure there is no disagreement doing this.
		if options!=None:
			if options.trace_level!=self.profiler_options.trace_level:
				raise error('In profiler session. Inconsistent trace level from run call')

		#TODO Process run options. Merge input run options with our ones.
		#There seem to be very few other relevant ones at the moment.
		
		self.local_run_metadata = tf.RunMetadata()
		super(TracerSessionAbstract, self).run(fetches, feed_dict=feed_dict, options=options, run_metadata=self.local_run_metadata)
		
		#This means run metadata was requested externally
		#so use set it from the one we passed through to tf.
		if run_metadata != None:
			run_metadata = self.local_run_metadata

		tl = timeline.Timeline(self.local_run_metadata.step_stats)
		ctf = tl.generate_chrome_trace_format()
		with open(self.getFileName(), 'w') as f:
			f.write(ctf)
		
		if self.eachTime:
			self.counter = self.counter + 1

#This pattern is a bit like an explicitly defined decorator.
#Allows TracerSession to have 
#exactly the same constructor as tf.Session.
def TracerSessionFactory(outputFileName, outputDirectory=None, eachTime=False):
	if (outputDirectory==None) and (eachTime==True):
		raise error("In profiler session. Must specify a directory to use eachTime mode.")	
	if (eachTime==True):
		print("Warning in profiler session. eachTime mode can produce large outputs.")		
	class TracerSessionDerived(TracerSessionAbstract):
		def getOutputParams(self):
			return outputFileName,outputDirectory,eachTime
	return TracerSessionDerived

class SessionFactory:
	def __init__(self,sessionClass):
		assert( issubclass( sessionClass, tf.Session ) )
		self._sessionClass = sessionClass

	def getSession(self):
		return self._sessionClass

	def setSessionClass(self,sessionClass):
		assert( issubclass( sessionClass, tf.Session ) )
		self._sessionClass = sessionClass

#A global object. Know what you are doing before changing this.
_sessionFactory = SessionFactory(tf.Session) 

def getSession():
	return _sessionFactory.getSession()

#Know what you are doing before calling this.
def setSession(sessionClass):
	_sessionFactory.setSessionClass(sessionClass)
