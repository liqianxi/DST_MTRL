import torch
import unittest
from nets import MaskGeneratorNet  # Import your neural network class

class TestDifferentiability(unittest.TestCase):
    def setUp(self):
        # Initialize your neural network
        self.net = MaskGeneratorNet()

    def test_forward_differentiable(self):
        # Define input tensors for testing
        input_tensor_2d = torch.randn((3, 4), requires_grad=True)
        input_tensor_1d = torch.randn(4, requires_grad=True)

        # Check differentiability for the 2d tensor input
        self.assertTrue(torch.autograd.gradcheck(self.net.forward, input_tensor_2d))

        # Check differentiability for the 1d tensor input
        self.assertTrue(torch.autograd.gradcheck(self.net.forward, input_tensor_1d))

if __name__ == '__main__':
    unittest.main()