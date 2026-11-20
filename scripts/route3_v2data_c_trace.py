"""Exact intersections of opaque textured planes, shared by C pinhole and lens renders."""
import numpy as np
import torch
import triton as tr
import triton.language as tl


@tr.jit
def intersect(rect, shift, ax, ay, az, bx, by, bz, layers: tl.constexpr, hole: tl.constexpr):
    """The last plane fills the camera. Iterate back-to-front to recover the first two hits."""
    first = tl.full(ax.shape, layers-1, tl.int32)
    second = tl.full(ax.shape, -1, tl.int32)
    distance = (tl.load(rect + (layers-1)*5) - az) / bz
    x = ax + bx*distance
    y = ay + by*distance
    for j in tl.static_range(layers-2, -1, -1):
        row = rect + j*5
        d = (tl.load(row)-az) / bz
        u = ax + bx*d - (shift if j == 0 else 0.)
        v = ay + by*d
        l = tl.load(row+1)
        r = tl.load(row+2)
        b = tl.load(row+3)
        t = tl.load(row+4)
        inside = (u>=l)&(u<r)&(v>=b)&(v<t)&(d>0)
        if hole and j == 0:
            s = (u-l)/(r-l)*3
            q = (v-b)/(t-b)*4
            inside = inside & ~((tl.abs(s-tl.floor(s)-.5)<.205)&(tl.abs(q-tl.floor(q)-.5)<.23))
        second = tl.where(inside, first, second)
        first = tl.where(inside, j, first)
        x = tl.where(inside, u, x)
        y = tl.where(inside, v, y)
    return first, second, x, y


@tr.jit
def bilinear(texture, texel, fx, fy):
    p00 = tl.load(texture+texel)
    p01 = tl.load(texture+texel+1)
    p10 = tl.load(texture+texel+256)
    p11 = tl.load(texture+texel+257)
    return (p00+(p01-p00)*fx)*(1-fy) + (p10+(p11-p10)*fx)*fy


@tr.jit
def color_lookup(textures, materials, ids, x, y):
    row = materials + ids*5
    scale = tl.load(row)
    u = x*scale + tl.load(row+1)
    v = y*scale + tl.load(row+2)
    x = (u-tl.floor(u))*255
    y = (v-tl.floor(v))*255
    ix = tl.minimum(x.to(tl.int32), 254)
    iy = tl.minimum(y.to(tl.int32), 254)
    addr = tl.load(row+3).to(tl.int32)*3*65536 + iy*256 + ix
    gain = tl.load(row+4)
    r = gain*bilinear(textures, addr, x-ix, y-iy)
    g = gain*bilinear(textures, addr+65536, x-ix, y-iy)
    b = gain*bilinear(textures, addr+131072, x-ix, y-iy)
    return r, g, b


@tr.jit
def pinhole(rect, poses, motions, textures, materials, ids, colors, peeled,
            layers: tl.constexpr, hole: tl.constexpr, focal: tl.constexpr,
            block: tl.constexpr, center: tl.constexpr):
    idx = tl.program_id(0)*block + tl.arange(0, block)
    frame = tl.program_id(1)
    ox = tl.full((block,), 0., tl.float32) + tl.load(poses+frame*3)
    oy = tl.full((block,), 0., tl.float32) + tl.load(poses+frame*3+1)
    oz = tl.full((block,), 0., tl.float32) + tl.load(poses+frame*3+2)
    shift = tl.load(motions+frame)
    if center:
        pixel = idx
        dx, dy = .5, .5
    else:
        pixel = idx//16
        subpixel = idx%16
        dx = (.5+subpixel%4)/4
        dy = (.5+subpixel//4)/4
    x = ((pixel%256).to(tl.float32)+dx-128)/focal
    y = ((pixel//256).to(tl.float32)+dy-128)/focal
    first, second, u, v = intersect(rect, shift, ox, oy, oz, x, y, 1., layers, hole)
    if center:
        pointer = peeled + (frame*65536+pixel)*2
        tl.store(pointer, first)
        tl.store(pointer+1, second)
    else:
        r, g, b = color_lookup(textures, materials, first, u, v)
        i = frame*65536*16+idx
        tl.store(ids+i, first)
        tl.store(colors+i*3, r)
        tl.store(colors+i*3+1, g)
        tl.store(colors+i*3+2, b)


@tr.jit
def optical(rect, textures, materials, pupil, out, ox, oy, oz, shift, focus,
            layers: tl.constexpr, hole: tl.constexpr, focal: tl.constexpr,
            pixels: tl.constexpr, samples: tl.constexpr):
    px = tl.program_id(0)*pixels + tl.arange(0, pixels)
    chunk = tl.program_id(1)
    phase = chunk//(256//samples)
    s = (chunk%(256//samples))*samples + tl.arange(0, samples)
    lens_x = tl.load(pupil+s*3)
    lens_y = tl.load(pupil+s*3+1)
    x = ((px%256).to(tl.float32) + (.5+phase%2)/2 - 128) / focal
    y = ((px//256).to(tl.float32) + (.5+phase//2)/2 - 128) / focal
    ax = tl.broadcast_to(ox+lens_x[None, :], (pixels, samples))
    ay = tl.broadcast_to(oy+lens_y[None, :], (pixels, samples))
    bx = x[:, None] - lens_x[None, :]/focus
    by = y[:, None] - lens_y[None, :]/focus
    first, _, u, v = intersect(rect, shift, ax, ay, oz, bx, by, 1., layers, hole)
    r, g, b = color_lookup(textures, materials, first, u, v)
    dst = out + (chunk*65536 + px)*6
    tl.store(dst, tl.sum(tl.minimum(r, 1.), axis=1))
    tl.store(dst+1, tl.sum(tl.minimum(g, 1.), axis=1))
    tl.store(dst+2, tl.sum(tl.minimum(b, 1.), axis=1))
    tl.store(dst+3, tl.sum(r, axis=1))
    tl.store(dst+4, tl.sum(g, axis=1))
    tl.store(dst+5, tl.sum(b, axis=1))


@tr.jit
def aperture_labels(rect, poses, motions, pupil, out_z, out_id, out_world, out_flags, out_witness, focus,
                    frame: tl.constexpr, frames: tl.constexpr, layers: tl.constexpr, hole: tl.constexpr,
                    focal: tl.constexpr, block: tl.constexpr):
    pixel = tl.program_id(0)*block + tl.arange(0, block)
    lens = tl.program_id(1)
    ox = tl.load(poses+frame*3)
    oy = tl.load(poses+frame*3+1)
    oz = tl.load(poses+frame*3+2)
    shift = tl.load(motions+frame)
    lx = tl.load(pupil+lens*3)
    ly = tl.load(pupil+lens*3+1)
    ax = tl.full((block,), 0., tl.float32)+ox+lx
    ay = tl.full((block,), 0., tl.float32)+oy+ly
    x = ((pixel%256).to(tl.float32)+.5-128)/focal
    y = ((pixel//256).to(tl.float32)+.5-128)/focal
    ids, _, u, v = intersect(rect, shift, ax, ay, oz, x-lx/focus, y-ly/focus, 1., layers, hole)
    zw = tl.load(rect+ids*5)
    distance = zw-oz
    wx = u+tl.where(ids==0, shift, 0.)
    sx = (wx-ox)/distance*focal+128
    sy = (v-oy)/distance*focal+128
    center_view = (sx>=0)&(sx<256)&(sy>=0)&(sy<256)
    center, _, _, _ = intersect(rect, shift, ax-lx, ay-ly, oz, (wx-ox)/distance, (v-oy)/distance,
                                 1., layers, hole)
    hidden = center_view & (center<ids)
    witness = tl.full((block,), 0, tl.int32)
    for j in range(frames):
        qx = tl.load(poses+j*3)
        qy = tl.load(poses+j*3+1)
        qz = tl.load(poses+j*3+2)
        move = tl.load(motions+j)
        xd = (u+tl.where(ids==0, move, 0.)-qx)/(zw-qz)
        yd = (v-qy)/(zw-qz)
        visible_id, _, _, _ = intersect(rect, move, ax*0.+qx, ay*0.+qy, qz, xd, yd, 1., layers, hole)
        px, py = xd*focal+128, yd*focal+128
        seen = (px>=0)&(px<256)&(py>=0)&(py<256)&(visible_id==ids)&(j!=frame)
        witness |= tl.where(hidden & seen, 1 << j, 0)
    index = lens*65536+pixel
    tl.store(out_z+index, distance)
    tl.store(out_id+index, ids+1)
    tl.store(out_world+index*3, wx)
    tl.store(out_world+index*3+1, v)
    tl.store(out_world+index*3+2, zw)
    tl.store(out_flags+index, center_view.to(tl.int32)+2*hidden.to(tl.int32))
    tl.store(out_witness+index, witness)


FOCAL = 256 * 50 / 36


def trace_pinhole(rect, poses, motion, textures, mats, hole):
    frames = len(poses)
    ids = torch.empty((frames, 256, 256, 16), dtype=torch.uint8, device="cuda")
    rgb = torch.empty((*ids.shape, 3), dtype=torch.float32, device="cuda")
    nearest = torch.empty((frames, 256, 256, 2), dtype=torch.int8, device="cuda")
    for center in (False, True):
        rays = 65536 if center else 65536*16
        pinhole[(rays//128, frames)](rect, poses, motion, textures, mats, ids, rgb, nearest,
                                    len(rect), hole, FOCAL, 128, center)
    return ids, rgb, nearest


def trace_optical(rect, textures, mats, pupil, center, shift, focus, hole):
    partials = torch.empty((16, 256, 256, 6), dtype=torch.float32, device="cuda")
    optical[(65536//8, 16)](rect, textures, mats, pupil, partials, *map(float, center), float(shift),
                              float(focus), len(rect), hole, FOCAL, 8, 64)
    result = (partials.sum(0)/1024).half().cpu().numpy()
    return result[..., :3], result[..., 3:]


def trace_aperture(rect, poses, motion, pupil, focus, middle, hole):
    shape = (len(pupil), 256, 256)
    depth = torch.empty(shape, dtype=torch.float16, device="cuda")
    world = torch.empty((*shape, 3), dtype=torch.float16, device="cuda")
    ids = torch.empty(shape, dtype=torch.uint8, device="cuda")
    flags = torch.empty_like(ids)
    witness = torch.empty(shape, dtype=torch.int16, device="cuda")
    aperture_labels[(65536//128, len(pupil))](rect, poses, motion, pupil, depth, ids, world, flags, witness,
                                           float(focus), middle, len(poses), len(rect), hole, FOCAL, 128)
    seen = witness.cpu().numpy()
    bits = flags.cpu().numpy()
    hidden = bits&2 != 0
    return dict(depth=depth.cpu().numpy(), surface_id=ids.cpu().numpy(), world=world.cpu().numpy(),
                light_valid=np.ones_like(hidden), center_in_view=bits&1 != 0, hidden=hidden,
                visible_source_frames=seen, recoverable=seen>0, never_seen=hidden&(seen==0))
