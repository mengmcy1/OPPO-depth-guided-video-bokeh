"""Explicit, instance-only ATB encoder residual scale=1 experiment."""
import ast
import inspect
import textwrap
import types


def candidate_forward(original):
    """Retain the installed implementation except its one final division."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = 0

    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal matches
            expected = ast.parse('output_tensor = (input_tensor + hidden_states) / self.output_scale_factor').body[0]
            if ast.dump(node, include_attributes=False) != ast.dump(expected, include_attributes=False):
                return self.generic_visit(node)
            matches += 1
            return ast.parse('output_tensor = input_tensor + hidden_states\n'
                             'if self.output_scale_factor != 1:\n'
                             '    output_tensor = output_tensor / self.output_scale_factor').body

    tree = Replace().visit(tree)
    assert matches == 1, 'Installed ResNet source differs: do not guess another implementation'
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<ATB_encoder_scale1_candidate>', 'exec'), namespace)
    return namespace[original.__name__]


class EncoderScale1:
    def __init__(self, pipe, enabled=False):
        from diffusers.models.resnet import ResnetBlock2D
        self.pipe, self.enabled, self.originals = pipe, enabled, []
        self.modules = [(name, module) for name, module in pipe.vae.encoder.named_modules()
                        if isinstance(module, ResnetBlock2D)]
        assert self.modules, 'No actual encoder ResNet modules found'
        original_class_forward = ResnetBlock2D.forward
        replacement = candidate_forward(original_class_forward) if enabled else None
        for name, module in self.modules:
            assert type(module.output_scale_factor) in (float, int)
            assert module.forward.__func__ is original_class_forward
            if enabled:
                self.originals.append((module, module.forward))
                module.forward = types.MethodType(replacement, module)
        assert ResnetBlock2D.forward is original_class_forward
        encoder_ids = {id(module) for _, module in self.modules}
        for root in [pipe.vae.decoder, pipe.unet]:
            for module in root.modules():
                assert id(module) not in encoder_ids
                if isinstance(module, ResnetBlock2D):
                    assert module.forward.__func__ is original_class_forward
        self.runtime_hits = []
        self.handles = []

    def observe_warmup(self):
        for name, module in self.modules:
            def observe(mod, args, name=name):
                self.runtime_hits.append(dict(module=name, scale=mod.output_scale_factor,
                    skip_division=self.enabled and mod.output_scale_factor == 1,
                    input_shape=list(args[0].shape), input_dtype=str(args[0].dtype)))
            self.handles.append(module.register_forward_pre_hook(observe))

    def stop_observing(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return dict(enabled=self.enabled, modules=len(self.modules), hits=self.runtime_hits,
            observed_modules=sorted({row['module'] for row in self.runtime_hits}),
            scope='Only this pipe.vae.encoder; decoder/UNet/class/site-packages unchanged')

    def restore(self):
        self.stop_observing()
        for module, original in self.originals:
            module.forward = original
