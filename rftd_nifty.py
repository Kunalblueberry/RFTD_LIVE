"""
RFTD Strategy - NIFTY
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.getenv('ROOT_DIR'))

from rftd_main import RFTDStrategy
from Utils.alerts import Alert

# Launch mode
launch_mode = 'cronjob' if len(sys.argv) > 1 else 'manual'

rftd = RFTDStrategy(
    strategy_name='RFTD_NIFTY',
    tag_name='RFTD_NIFTY',
    symbol='NIFTY',
    launch_mode=launch_mode
)
rftd.execute()