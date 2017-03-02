import GPflow
import simple_regression

GPflow.profiler.activateTracer('timeline_last')
simple_regression.runExperiments(sampling=False)
GPflow.profiler.deactivateTracer()

GPflow.profiler.activateTracer('timeline',outputDirectory='profiler_output',eachTime=True)
simple_regression.runExperiments(sampling=False)
GPflow.profiler.deactivateTracer()
