"""
LLM Signal Fusion Module for Trading Signals
Fast local LLM integration using Ollama for signal enhancement
"""

import json
import time
import logging
import ollama
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


def load_llm_config_from_json(config_file: str = 'config.json') -> 'LLMConfig':
    """Load LLM configuration from existing config.json file"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            llm_settings = config_data.get('llm_config', {})
            
            return LLMConfig(
                model_name=llm_settings.get('model_name', 'qwen2.5:3b-instruct-q4_K_M'),
                temperature=llm_settings.get('temperature', 0.1),
                max_tokens=llm_settings.get('max_tokens', 256),
                timeout_ms=llm_settings.get('timeout_ms', 6000000),
                min_confidence=llm_settings.get('min_confidence', 0.6),
                enabled=llm_settings.get('enabled', True),
                ollama_host=llm_settings.get('ollama_host', 'localhost:11434'),
                num_parallel=llm_settings.get('num_parallel', 4),
                max_loaded_models=llm_settings.get('max_loaded_models', 3)
            )
        else:
            logging.getLogger(__name__).warning(f"Config file {config_file} not found, using defaults")
            return LLMConfig()
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading LLM config from {config_file}: {e}")
        return LLMConfig()


@dataclass
class LLMConfig:
    """Configuration for LLM signal fusion"""
    model_name: str = 'qwen2.5:3b-instruct-q4_K_M'  # Ultra-fast 0.6B model
    temperature: float = 0.1
    max_tokens: int = 256
    timeout_ms: int = 60000000  # Max inference time
    min_confidence: float = 0.6
    enabled: bool = True
    ollama_host: str = 'localhost:11434'
    num_parallel: int = 4
    max_loaded_models: int = 3


class LLMSignalFusion:
    """
    Fast LLM-based signal fusion for crypto trading
    Combines traditional technical indicators with LLM analysis
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, config_file: str = 'config.json'):
        self.config = config or load_llm_config_from_json(config_file)
        self.logger = logging.getLogger(__name__)
        self._model_warmed = False
        self._last_inference_time = 0
        
        if self.config.enabled:
            self._initialize_ollama()
        
        self.logger.info(f"LLM Config loaded: {self.config.model_name}, enabled: {self.config.enabled}")
    
    def _initialize_ollama(self):
        """Initialize and warm up Ollama model"""
        try:
            # Set Ollama host if specified
            if hasattr(ollama, '_client') and self.config.ollama_host != 'localhost:11434':
                ollama._client.base_url = f"http://{self.config.ollama_host}"
            
            # Check if model exists
            models = ollama.list()
            model_exists = any(self.config.model_name in model['name'] for model in models.get('models', []))
            
            if not model_exists:
                self.logger.warning(f"Model {self.config.model_name} not found. Please run: ollama pull {self.config.model_name}")
                self.config.enabled = False
                return
            
            # Warm up model
            self._warmup_model()
            self.logger.info(f"LLM Signal Fusion initialized with {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {e}")
            self.config.enabled = False
    
    def _warmup_model(self):
        """Pre-warm model for faster inference"""
        try:
            ollama.chat(
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': 'ready'}],
                options={'num_predict': 1}
            )
            self._model_warmed = True
            self.logger.debug(f"Model {self.config.model_name} warmed up")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def get_enhanced_signal(self, market_data: Dict, traditional_signals: Dict) -> Tuple[str, Dict]:
        """
        Get enhanced trading signal using LLM fusion
        
        Args:
            market_data: Current market data (price, volume, etc.)
            traditional_signals: Signals from technical indicators
            
        Returns:
            Tuple of (final_signal, enhanced_indicators)
        """
        if not self.config.enabled:
            return self._fallback_signal(traditional_signals)
        
        try:
            # Prepare market context
            market_context = self._prepare_market_context(market_data, traditional_signals)
            
            # Get LLM analysis
            llm_analysis = self._get_llm_analysis(market_context)
            
            # Combine signals intelligently
            final_signal = self._combine_signals(traditional_signals, llm_analysis)
            
            # Enhanced indicators
            enhanced_indicators = traditional_signals.copy()
            enhanced_indicators.update({
                'llm_signal': llm_analysis.get('signal', 'none'),
                'llm_confidence': llm_analysis.get('confidence', 0.0),
                'llm_reasoning': llm_analysis.get('reasoning', ''),
                'signal_fusion': final_signal,
                'inference_time_ms': self._last_inference_time
            })
            
            return final_signal, enhanced_indicators
            
        except Exception as e:
            self.logger.error(f"LLM signal fusion error: {e}")
            return self._fallback_signal(traditional_signals)
    
    def _prepare_market_context(self, market_data: Dict, traditional_signals: Dict) -> Dict:
        """Prepare structured market context for LLM"""
        
        # Extract current price and trends
        current_price = market_data.get('current_price', 0)
        price_change = market_data.get('price_change_24h', 0)
        volume_trend = market_data.get('volume_trend', 'neutral')
        
        # Count signal consensus
        signals = []
        for strategy, signal in traditional_signals.items():
            if isinstance(signal, str) and signal in ['buy', 'sell', 'none']:
                signals.append(signal)
        
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        none_count = signals.count('none')
        
        # Extract key indicators
        indicators = self._extract_indicators(traditional_signals)
        
        return {
            'price': current_price,
            'price_change_24h': price_change,
            'volume_trend': volume_trend,
            'signal_consensus': {
                'buy': buy_count,
                'sell': sell_count,
                'none': none_count,
                'total': len(signals)
            },
            'indicators': indicators,
            'market_regime': self._determine_market_regime(indicators)
        }
    
    def _extract_indicators(self, traditional_signals: Dict) -> Dict:
        """Extract key indicator values from traditional signals"""
        indicators = {}
        
        # RSI
        if 'rsi' in traditional_signals:
            indicators['rsi'] = traditional_signals['rsi']
        
        # MACD
        if 'macd_line' in traditional_signals:
            indicators['macd_line'] = traditional_signals['macd_line']
            indicators['macd_signal'] = traditional_signals.get('macd_signal', 0)
            indicators['macd_histogram'] = indicators['macd_line'] - indicators['macd_signal']
        
        # Supertrend
        if 'st_direction' in traditional_signals:
            indicators['supertrend_direction'] = traditional_signals['st_direction']
            indicators['supertrend_bullish'] = traditional_signals['st_direction'] == 1
        
        # QQE
        if 'qqe_value' in traditional_signals:
            indicators['qqe_value'] = traditional_signals['qqe_value']
            indicators['qqe_signal'] = traditional_signals.get('qqe_signal', 0)
            indicators['qqe_bullish'] = indicators['qqe_value'] > indicators['qqe_signal']
        
        # TSI
        if 'tsi_line' in traditional_signals:
            indicators['tsi_line'] = traditional_signals['tsi_line']
            indicators['tsi_signal'] = traditional_signals.get('tsi_signal', 0)
            indicators['tsi_bullish'] = indicators['tsi_line'] > indicators['tsi_signal']
        
        # VWAP
        if 'vwap' in traditional_signals:
            indicators['vwap'] = traditional_signals['vwap']
            current_price = traditional_signals.get('current_price', 0)
            indicators['price_vs_vwap'] = 'above' if current_price > indicators['vwap'] else 'below'
        
        # ROC (multi-timeframe)
        if 'current_roc_3m' in traditional_signals:
            indicators['roc_3m'] = traditional_signals['current_roc_3m']
            indicators['roc_15m'] = traditional_signals.get('current_roc_15m', 0)
            indicators['roc_divergence'] = abs(indicators['roc_3m'] - indicators['roc_15m'])
        
        return indicators
    
    def _determine_market_regime(self, indicators: Dict) -> str:
        """Determine current market regime"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Check various indicators
        if indicators.get('rsi', 50) > 60:
            bullish_signals += 1
        elif indicators.get('rsi', 50) < 40:
            bearish_signals += 1
        
        if indicators.get('macd_histogram', 0) > 0:
            bullish_signals += 1
        elif indicators.get('macd_histogram', 0) < 0:
            bearish_signals += 1
        
        if indicators.get('supertrend_bullish', False):
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if indicators.get('qqe_bullish', False):
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_llm_analysis(self, market_context: Dict) -> Dict:
        """Get LLM analysis with structured output"""
        
        prompt = self._build_trading_prompt(market_context)
        
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={
                    'temperature': self.config.temperature,
                    'num_ctx': 4096,
                    'num_predict': self.config.max_tokens,
                    'top_k': 40,
                    'top_p': 0.95
                }
            )
            
            self._last_inference_time = (time.time() - start_time) * 1000
            
            if self._last_inference_time > self.config.timeout_ms:
                self.logger.warning(f"LLM inference slow: {self._last_inference_time:.1f}ms")
            
            result = json.loads(response['message']['content'])
            return self._validate_llm_output(result)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"LLM JSON parse error: {e}")
            return {'signal': 'none', 'confidence': 0.0, 'reasoning': 'JSON parse error'}
        
        except Exception as e:
            self.logger.error(f"LLM inference error: {e}")
            return {'signal': 'none', 'confidence': 0.0, 'reasoning': f'Error: {str(e)}'}
    
    def _build_trading_prompt(self, market_context: Dict) -> str:
        """Build structured prompt for LLM analysis"""
        
        indicators = market_context['indicators']
        consensus = market_context['signal_consensus']
        
        return f"""
You are an expert crypto trader analyzing market conditions for a trading signal.

MARKET DATA:
- Price: ${market_context['price']:.6f}
- 24h Change: {market_context['price_change_24h']:.2f}%
- Volume Trend: {market_context['volume_trend']}
- Market Regime: {market_context['market_regime']}

TECHNICAL INDICATORS:
- RSI: {indicators.get('rsi', 'N/A')} {'(Overbought)' if indicators.get('rsi', 50) > 70 else '(Oversold)' if indicators.get('rsi', 50) < 30 else '(Neutral)'}
- MACD: {indicators.get('macd_histogram', 'N/A')} {'(Bullish)' if indicators.get('macd_histogram', 0) > 0 else '(Bearish)'}
- Supertrend: {'Bullish' if indicators.get('supertrend_bullish', False) else 'Bearish'}
- QQE: {'Bullish' if indicators.get('qqe_bullish', False) else 'Bearish'}
- TSI: {'Bullish' if indicators.get('tsi_bullish', False) else 'Bearish'}
- VWAP: Price is {indicators.get('price_vs_vwap', 'unknown')} VWAP
- ROC 3m: {indicators.get('roc_3m', 'N/A')}
- ROC 15m: {indicators.get('roc_15m', 'N/A')}

STRATEGY CONSENSUS:
- BUY signals: {consensus['buy']}/{consensus['total']}
- SELL signals: {consensus['sell']}/{consensus['total']}
- NONE signals: {consensus['none']}/{consensus['total']}

Analyze this data and provide a trading recommendation.

Return ONLY valid JSON:
{{
    "signal": "buy|sell|none",
    "confidence": 0.0-1.0,
    "reasoning": "brief technical analysis",
    "timeframe": "short|medium|long",
    "risk_level": "low|medium|high"
}}
"""
    
    def _validate_llm_output(self, result: Dict) -> Dict:
        """Validate and sanitize LLM output"""
        
        # Validate signal
        signal = result.get('signal', 'none').lower()
        if signal not in ['buy', 'sell', 'none']:
            signal = 'none'
        
        # Validate confidence
        confidence = float(result.get('confidence', 0.0))
        confidence = max(0.0, min(1.0, confidence))
        
        # Validate other fields
        reasoning = str(result.get('reasoning', ''))[:200]  # Limit length
        timeframe = result.get('timeframe', 'short')
        risk_level = result.get('risk_level', 'medium')
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'timeframe': timeframe,
            'risk_level': risk_level
        }
    
    def _combine_signals(self, traditional_signals: Dict, llm_analysis: Dict) -> str:
        """Intelligent signal combination logic"""
        
        # Extract traditional signal consensus
        signals = []
        for strategy, signal in traditional_signals.items():
            if isinstance(signal, str) and signal in ['buy', 'sell', 'none']:
                signals.append(signal)
        
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        none_count = signals.count('none')
        total_signals = len(signals)
        
        llm_signal = llm_analysis.get('signal', 'none')
        llm_confidence = llm_analysis.get('confidence', 0.0)
        
        # High-confidence LLM override
        if llm_confidence >= 0.85:
            self.logger.debug(f"High-confidence LLM override: {llm_signal} ({llm_confidence:.2f})")
            return llm_signal
        
        # Strong consensus + LLM agreement
        if buy_count >= (total_signals * 0.7) and llm_signal == 'buy' and llm_confidence >= self.config.min_confidence:
            return 'buy'
        elif sell_count >= (total_signals * 0.7) and llm_signal == 'sell' and llm_confidence >= self.config.min_confidence:
            return 'sell'
        
        # Moderate consensus + LLM confirmation
        if buy_count > sell_count and llm_signal == 'buy' and llm_confidence >= self.config.min_confidence:
            return 'buy'
        elif sell_count > buy_count and llm_signal == 'sell' and llm_confidence >= self.config.min_confidence:
            return 'sell'
        
        # LLM tiebreaker for close consensus
        if abs(buy_count - sell_count) <= 1 and llm_confidence >= 0.75:
            return llm_signal
        
        # Fallback to traditional consensus
        if buy_count > sell_count:
            return 'buy'
        elif sell_count > buy_count:
            return 'sell'
        else:
            return 'none'
    
    def _fallback_signal(self, traditional_signals: Dict) -> Tuple[str, Dict]:
        """Fallback when LLM is disabled or fails"""
        
        signals = []
        for strategy, signal in traditional_signals.items():
            if isinstance(signal, str) and signal in ['buy', 'sell', 'none']:
                signals.append(signal)
        
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count > sell_count:
            final_signal = 'buy'
        elif sell_count > buy_count:
            final_signal = 'sell'
        else:
            final_signal = 'none'
        
        enhanced_indicators = traditional_signals.copy()
        enhanced_indicators.update({
            'llm_signal': 'disabled',
            'llm_confidence': 0.0,
            'signal_fusion': final_signal
        })
        
        return final_signal, enhanced_indicators


# Singleton instance for global use
_llm_fusion_instance = None

def get_llm_fusion(config: Optional[LLMConfig] = None, config_file: str = 'config.json') -> LLMSignalFusion:
    """Get singleton LLM fusion instance"""
    global _llm_fusion_instance
    if _llm_fusion_instance is None:
        _llm_fusion_instance = LLMSignalFusion(config, config_file)
    return _llm_fusion_instance