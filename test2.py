import torch
import torch.nn.functional as F
import numpy as np

def format_c_array(data, name: str, dtype: str = "float", items_per_line: int = 8):
    """Formats a numpy array into a C-style array string."""
    header = f"{dtype} {name}[] = {{\n    "
    body = []
    
    flat_data = data.flatten()
    for i, val in enumerate(flat_data):
        if dtype == "float":
            body.append(f"{val:.6f}f")
        else:
            body.append(str(val))
        
        if i < len(flat_data) - 1:
            body.append(", ")
        
        if (i + 1) % items_per_line == 0 and i < len(flat_data) - 1:
            body.append("\n    ")
            
    footer = "\n};\n"
    return header + "".join(body) + footer

def generate_backward_test_cases(filename="softmax_backward_cases.txt"):
    """
    Generates test cases for the GradFn_softmax BACKWARD pass.
    """
    # Define shapes to test (1D, 2D, 3D, 4D)
    test_shapes = [
        (6,),
        (4, 5),
        (3, 4, 5),
        (2, 3, 4, 5)
    ]

    with open(filename, 'w') as fp:
        fp.write("// Auto-generated test cases for GradFn_softmax (BACKWARD PASS)\n\n")
        
        test_case_count = 0
        for shape in test_shapes:
            for dim in range(len(shape)):
                test_case_count += 1
                
                fp.write(f"// --- Test Case {test_case_count}: Shape={list(shape)}, Dim={dim} ---\n")

                # Create input tensor that requires gradients
                a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
                
                # Perform forward pass
                softmax_output = F.softmax(a, dim=dim)
                
                # Perform backward pass with a standard upstream gradient of ones
                upstream_grad = torch.ones_like(softmax_output)
                softmax_output.backward(gradient=upstream_grad)
                
                # a.grad now contains the final gradient (dL/da) we want to verify
                expected_input_grad = a.grad
                
                # Write data to file
                c_shape = list(shape) + [0] * (4 - len(shape)) # Pad to 4D for C struct
                fp.write(f"TensorShape shape_{test_case_count} = {{ {', '.join(map(str, c_shape))} }};\n")
                fp.write(f"int dim_{test_case_count} = {dim};\n")

                # --- THIS IS THE FIX ---
                # Save the original input data (the logits) which is needed for the C test setup.
                fp.write(format_c_array(a.detach().numpy(), f"input_data_{test_case_count}"))

                # This data is still useful for reference but the C test primarily needs the input_data above.
                fp.write(format_c_array(softmax_output.detach().numpy(), f"softmax_output_data_{test_case_count}"))
                fp.write(format_c_array(upstream_grad.numpy(), f"upstream_grad_data_{test_case_count}"))
                fp.write(format_c_array(expected_input_grad.numpy(), f"expected_grad_{test_case_count}"))

                fp.write("\n" + "-"*80 + "\n\n")
                print(f"Generated BACKWARD test case for shape={list(shape)}, dim={dim}")

if __name__ == '__main__':
    generate_backward_test_cases()
    print("\n✅ Successfully generated backward pass test cases in 'softmax_backward_cases.txt'")