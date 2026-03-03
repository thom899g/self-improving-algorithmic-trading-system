"""
Main orchestrator for the self-improving trading system.
Architectural choice: Centralized orchestration with dependency injection
to allow swapping components without system-wide changes.
"""
import logging
import sys
from typing import Dict, Any
from datetime import datetime
import traceback

from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from trading_environment import TradingEnvironment
from rl_agent import RLAgent
from risk_manager import RiskManager
from firebase_manager import FirebaseManager
from performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'trading_system_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system orchestrator with failure recovery."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize all system components with dependency injection."""
        self.config = config
        self.is_running = False
        self.components = {}
        
        try:
            # Initialize components in dependency order
            logger.info("Initializing Firebase manager...")
            self.firebase = FirebaseManager(config)
            self.components['firebase'] = self.firebase
            
            logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(config, self.firebase)
            self.components['risk_manager'] = self.risk_manager
            
            logger.info("Initializing data collector...")
            self.data_collector = DataCollector(config, self.firebase)
            self.components['data_collector'] = self.data_collector
            
            logger.info("Initializing data preprocessor...")
            self.data_preprocessor = DataPreprocessor(config)
            self.components['data_preprocessor'] = self.data_preprocessor
            
            logger.info("Initializing trading environment...")
            self.environment = TradingEnvironment(config)
            self.components['environment'] = self.environment
            
            logger.info("Initializing RL agent...")
            self.agent = RLAgent(config, self.firebase)
            self.components['agent'] = self.agent
            
            logger.info("Initializing performance monitor...")
            self.performance_monitor = PerformanceMonitor(config, self.firebase)
            self.components['performance_monitor'] = self.performance_monitor
            
            # Register system state in Firebase
            self.firebase.update_system_state({
                'status': 'initialized',
                'last_updated': datetime.utcnow().isoformat(),
                'components': list(self.components.keys())
            })
            
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {str(e)}")
            self._handle_critical_failure(e)
            raise
    
    def _handle_critical_failure(self, error: Exception) -> None:
        """Handle critical failures with emergency notification."""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log to Firebase for post-mortem analysis
        self.firebase.log_error(error_details)
        
        # Send emergency notification via Telegram
        try:
            self.firebase.send_emergency_alert(f"CRITICAL SYSTEM FAILURE: {error_details['error_type']}")
        except:
            logger.critical("Failed to send emergency notification")
    
    def run_training_cycle(self) -> bool:
        """Execute one complete training cycle with error handling."""
        try:
            if not self.risk_manager.check_trading_allowed():
                logger.warning("Trading suspended by risk manager")
                return False
            
            # 1. Collect latest market data
            logger.info("Collecting market data...")
            raw_data = self.data_collector.fetch_latest_data()
            
            if raw_data.empty:
                logger.warning("No data collected, skipping cycle")
                return False
            
            # 2. Preprocess data
            logger.info("Preprocessing data...")
            processed_data = self.data_preprocessor.transform(raw_data)
            
            # 3. Update environment state
            logger.info("Updating environment...")
            state = self.environment.update_state(processed