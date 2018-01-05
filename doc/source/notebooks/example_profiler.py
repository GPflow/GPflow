import gpflow
import simple_regression

gpflow.profiler.activateTracer('timeline_last')
simple_regression.runExperiments(sampling=False)
gpflow.profiler.deactivateTracer()

gpflow.profiler.activateTracer('timeline', outputDirectory='profiler_output', eachTime=True)
simple_regression.runExperiments(sampling=False)
gpflow.profiler.deactivateTracer()
