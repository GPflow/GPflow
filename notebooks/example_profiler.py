import GPflow
import simple_regression

GPflow.profiler.activateTracer('timeline_last')
simple_regression.runExperiments(sampling=False)
GPflow.profiler.deactivateTracer()

GPflow.profiler.activateTracer('timeline', outputDirectory='profiler_output', each_time=True)
simple_regression.runExperiments(sampling=False)
GPflow.profiler.deactivateTracer()
