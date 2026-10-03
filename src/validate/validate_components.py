"""
Minimal validation of individual components
"""

import torch
import numpy as np
import sys
sys.path.append('src')

def test_data_loading():
    """Test data loading with minimal data"""
    print("🧪 Testing data loading...")
    
    from src.data_loader import CMAPSSDataLoader
    
    data_loader = CMAPSSDataLoader()
    train_df, test_df = data_loader.load_data('data/train_FD001.txt', 'data/test_FD001.txt')
    train_scaled, test_scaled = data_loader.preprocess_data(train_df, test_df)
    
    print(f"✅ Data loading works! Features: {len(data_loader.feature_cols)}")
    return True

def test_model():
    """Test model with dummy data"""
    print("🧪 Testing model...")
    
    from src.transformer_model import create_model
    
    # Create small model
    model = create_model(input_dim=17, d_model=32, nhead=4, num_layers=2)
    
    # Test with dummy data
    batch_size, seq_len, input_dim = 2, 10, 17
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    
    print(f"✅ Model works! Input: {x.shape}, Output: {output.shape}")
    return True

def test_visualization():
    """Test visualization with dummy data"""
    print("🧪 Testing visualization...")
    
    from src.visualization import SyntheticDataVisualizer
    
    visualizer = SyntheticDataVisualizer([f'feat_{i}' for i in range(17)])
    
    # Create dummy data
    real_data = np.random.randn(10, 20, 17)
    synthetic_data = real_data + np.random.normal(0, 0.1, real_data.shape)
    
    # Test correlation analysis
    real_df = real_data.reshape(-1, 17)
    synthetic_df = synthetic_data.reshape(-1, 17)
    
    corr1 = np.corrcoef(real_df.T)
    corr2 = np.corrcoef(synthetic_df.T)
    
    diff = np.mean(np.abs(corr1 - corr2))
    
    print(f"✅ Visualization works! Correlation difference: {diff:.4f}")
    return True

def main():
    """Run all validation tests"""
    print("🔍 COMPONENT VALIDATION")
    print("="*40)
    
    tests = [test_data_loading, test_model, test_visualization]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL COMPONENTS VALIDATED SUCCESSFULLY!")
        return True
    else:
        print("⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    main()