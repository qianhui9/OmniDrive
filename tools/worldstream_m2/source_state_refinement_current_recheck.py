"""Fresh final-code full-task checks; model weights load once for both inputs."""
import sys
from . import source_state_refinement_integration as integration


def main():
    model,dataset=integration.load()
    integration.load=lambda:(model,dataset)
    root='outputs/module1_source_state_refinement/gpu_validation_v2'
    for case,folder in (('natural','current_source_recheck'),('repeat','current_repeat_recheck')):
        sys.argv=['final-source-recheck','--case',case,'--scenes','1','--frames','4',
            '--warmup','0','--repeats','1','--replay-protocol',
            'outputs/module1_source_state_refinement/replay/protocol.json','--output',root+'/'+folder]
        if case=='repeat':sys.argv+=['--lease-experiment','--selective-probe']
        integration.main()


if __name__=='__main__':main()
