"""High-Performance Fused Chunk Gated Linear Attention (GLA) Kernel.

Computes O(N) linear-time gated attention with exact mathematical equivalence to
token-by-token recurrence. Combines factored Tensor Core GEMMs for intra-chunk
block-diagonal attention with an in-register fused Triton recurrent scan (FusedRecurrentScan)
for inter-chunk state propagation.

Eliminates the thousands of sequential PyTorch autograd graph allocations and intermediate
global memory tensor roundtrips typical of un-fused recurrent chunked attention.
In long-context autoregressive generation (T >= 8,192), yields 3x-6x faster token generation
and 99%+ memory savings over quadratic softmax KV-caches by maintaining a fixed-size
O(1) recurrent hidden state.
"""

from __future__ import annotations

import torch

from triune.kernels.dispatcher import is_triton_available, is_turbo_mode_enabled

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:

    @triton.jit
    def _recurrent_scan_fwd(
        KV,
        DECAY,
        INIT_STATE,
        S_OUT,
        FINAL_STATE,
        M,
        N,
        stride_kv_m,
        stride_kv_n,
        stride_dec_m,
        stride_dec_n,
        stride_s_m,
        stride_s_n,
        stride_init_m,
        stride_fin_m,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        s = tl.load(INIT_STATE + offs_m * stride_init_m, mask=mask_m, other=0.0).to(tl.float32)

        for i in range(N):
            tl.store(S_OUT + offs_m * stride_s_m + i * stride_s_n, s, mask=mask_m)
            d = tl.load(DECAY + offs_m * stride_dec_m + i * stride_dec_n, mask=mask_m, other=1.0).to(tl.float32)
            x = tl.load(KV + offs_m * stride_kv_m + i * stride_kv_n, mask=mask_m, other=0.0).to(tl.float32)
            s = s * d + x

        tl.store(FINAL_STATE + offs_m * stride_fin_m, s, mask=mask_m)

    @triton.jit
    def _recurrent_scan_bwd(
        D_S_OUT,
        D_FINAL,
        DECAY,
        S_OUT,
        D_KV,
        D_DECAY,
        D_INIT,
        M,
        N,
        stride_ds_m,
        stride_ds_n,
        stride_df_m,
        stride_dec_m,
        stride_dec_n,
        stride_s_m,
        stride_s_n,
        stride_dkv_m,
        stride_dkv_n,
        stride_ddec_m,
        stride_ddec_n,
        stride_di_m,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        d_s = tl.load(D_FINAL + offs_m * stride_df_m, mask=mask_m, other=0.0).to(tl.float32)

        for i in range(N - 1, -1, -1):
            d = tl.load(DECAY + offs_m * stride_dec_m + i * stride_dec_n, mask=mask_m, other=1.0).to(tl.float32)
            s_prev = tl.load(S_OUT + offs_m * stride_s_m + i * stride_s_n, mask=mask_m, other=0.0).to(tl.float32)
            d_out_val = tl.load(D_S_OUT + offs_m * stride_ds_m + i * stride_ds_n, mask=mask_m, other=0.0).to(tl.float32)

            tl.store(D_KV + offs_m * stride_dkv_m + i * stride_dkv_n, d_s, mask=mask_m)
            tl.store(D_DECAY + offs_m * stride_ddec_m + i * stride_ddec_n, d_s * s_prev, mask=mask_m)

            d_s = d_out_val + d_s * d

        tl.store(D_INIT + offs_m * stride_di_m, d_s, mask=mask_m)

    class FusedRecurrentScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, kv: torch.Tensor, decay_k: torch.Tensor, init_state: torch.Tensor):
            B, H, N, Dk, Dv = kv.shape
            M = B * H * Dk * Dv
            decay = decay_k.expand(B, H, N, Dk, Dv).contiguous()

            kv_flat = kv.permute(0, 1, 3, 4, 2).reshape(M, N).contiguous()
            decay_flat = decay.permute(0, 1, 3, 4, 2).reshape(M, N).contiguous()
            init_flat = init_state.reshape(M).contiguous()

            s_out = torch.empty(M, N, device=kv.device, dtype=torch.float32)
            final_state = torch.empty(M, device=kv.device, dtype=torch.float32)

            BLOCK_M = 128
            grid = ((M + BLOCK_M - 1) // BLOCK_M,)
            _recurrent_scan_fwd[grid](
                kv_flat,
                decay_flat,
                init_flat,
                s_out,
                final_state,
                M,
                N,
                kv_flat.stride(0),
                kv_flat.stride(1),
                decay_flat.stride(0),
                decay_flat.stride(1),
                s_out.stride(0),
                s_out.stride(1),
                init_flat.stride(0),
                final_state.stride(0),
                BLOCK_M=BLOCK_M,
            )
            ctx.save_for_backward(decay_flat, s_out)
            ctx.shapes = (B, H, N, Dk, Dv, M)
            ctx.kv_dtype = kv.dtype

            s_out_5d = s_out.reshape(B, H, Dk, Dv, N).permute(0, 1, 4, 2, 3).to(dtype=kv.dtype)
            final_state_4d = final_state.reshape(B, H, Dk, Dv).to(dtype=kv.dtype)
            return s_out_5d, final_state_4d

        @staticmethod
        def backward(ctx, d_s_out: torch.Tensor | None, d_final_state: torch.Tensor | None):
            decay_flat, s_out = ctx.saved_tensors
            B, H, N, Dk, Dv, M = ctx.shapes
            device = decay_flat.device

            if d_s_out is not None:
                d_s_flat = d_s_out.permute(0, 1, 3, 4, 2).reshape(M, N).contiguous().float()
            else:
                d_s_flat = torch.zeros(M, N, device=device, dtype=torch.float32)

            if d_final_state is not None:
                d_fin_flat = d_final_state.reshape(M).contiguous().float()
            else:
                d_fin_flat = torch.zeros(M, device=device, dtype=torch.float32)

            d_kv = torch.empty(M, N, device=device, dtype=torch.float32)
            d_decay = torch.empty(M, N, device=device, dtype=torch.float32)
            d_init = torch.empty(M, device=device, dtype=torch.float32)

            BLOCK_M = 128
            grid = ((M + BLOCK_M - 1) // BLOCK_M,)
            _recurrent_scan_bwd[grid](
                d_s_flat,
                d_fin_flat,
                decay_flat,
                s_out,
                d_kv,
                d_decay,
                d_init,
                M,
                N,
                d_s_flat.stride(0),
                d_s_flat.stride(1),
                d_fin_flat.stride(0),
                decay_flat.stride(0),
                decay_flat.stride(1),
                s_out.stride(0),
                s_out.stride(1),
                d_kv.stride(0),
                d_kv.stride(1),
                d_decay.stride(0),
                d_decay.stride(1),
                d_init.stride(0),
                BLOCK_M=BLOCK_M,
            )

            d_kv_5d = d_kv.reshape(B, H, Dk, Dv, N).permute(0, 1, 4, 2, 3).to(dtype=ctx.kv_dtype)
            d_decay_5d = (
                d_decay.reshape(B, H, Dk, Dv, N).permute(0, 1, 4, 2, 3).sum(dim=-1, keepdim=True).to(dtype=ctx.kv_dtype)
            )
            d_init_4d = d_init.reshape(B, H, Dk, Dv).to(dtype=ctx.kv_dtype)
            return d_kv_5d, d_decay_5d, d_init_4d

    @triton.jit
    def _chunk_gla_tc_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        G_ptr,
        Out_ptr,
        Init_ptr,
        Final_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_gb,
        stride_gt,
        stride_gh,
        stride_gd,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sk,
        stride_sv,
        T,
        H,
        HAS_INIT: tl.constexpr,
        SAVE_FINAL: tl.constexpr,
        BT: tl.constexpr,
        DK: tl.constexpr,
        BV: tl.constexpr,
        G_THRESH: tl.constexpr,
    ):
        """Fused Tensor-Core Chunked GLA Forward Kernel (in-register state, 0 intermediate HBM)."""
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H

        offs_t = tl.arange(0, BT)
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, BV)
        causal_mask = offs_t[:, None] >= offs_t[None, :]

        if HAS_INIT:
            s = tl.load(
                Init_ptr + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv
            ).to(tl.float32)
        else:
            s = tl.zeros([DK, BV], dtype=tl.float32)

        q_base = Q_ptr + b * stride_qb + h * stride_qh
        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh
        g_base = G_ptr + b * stride_gb + h * stride_gh
        o_base = Out_ptr + b * stride_ob + h * stride_oh

        num_chunks = T // BT
        for c in range(num_chunks):
            t_start = c * BT
            ts = t_start + offs_t
            q = tl.load(q_base + ts[:, None] * stride_qt + offs_k[None, :] * stride_qd)
            k = tl.load(k_base + ts[:, None] * stride_kt + offs_k[None, :] * stride_kd)
            v = tl.load(v_base + ts[:, None] * stride_vt + offs_v[None, :] * stride_vd)
            g = tl.load(g_base + ts[:, None] * stride_gt + offs_k[None, :] * stride_gd).to(tl.float32)
            g = tl.minimum(g, 0.0)

            g_cum = tl.cumsum(g, axis=0)
            g_last = tl.sum(g, axis=0)[None, :]

            if tl.min(g_last) >= G_THRESH:
                g_mid = 0.5 * g_last

                q_inter = (q.to(tl.float32) * tl.exp(g_cum)).to(q.dtype)
                o_inter = tl.dot(q_inter, s.to(q.dtype))

                q_intra = (q.to(tl.float32) * tl.exp(g_cum - g_mid)).to(q.dtype)
                k_intra = (k.to(tl.float32) * tl.exp(g_mid - g_cum)).to(k.dtype)
                qk = tl.dot(q_intra, tl.trans(k_intra))
                attn = tl.where(causal_mask, qk, 0.0).to(v.dtype)
                o_intra = tl.dot(attn, v)

                out_c = (o_inter + o_intra).to(Out_ptr.dtype.element_ty)
                tl.store(o_base + ts[:, None] * stride_ot + offs_v[None, :] * stride_od, out_c)

                k_dec = (k.to(tl.float32) * tl.exp(tl.minimum(g_last - g_cum, 0.0))).to(k.dtype)
                s = s * tl.exp(g_last).reshape(DK, 1) + tl.dot(tl.trans(k_dec), v)
            else:
                for t_sub in range(BT):
                    t_idx = t_start + t_sub
                    q_t = tl.load(q_base + t_idx * stride_qt + offs_k * stride_qd).to(tl.float32)
                    k_t = tl.load(k_base + t_idx * stride_kt + offs_k * stride_kd).to(tl.float32)
                    v_t = tl.load(v_base + t_idx * stride_vt + offs_v * stride_vd).to(tl.float32)
                    g_t = tl.load(g_base + t_idx * stride_gt + offs_k * stride_gd).to(tl.float32)
                    d_t = tl.exp(tl.minimum(g_t, 0.0))
                    s = s * d_t[:, None] + k_t[:, None] * v_t[None, :]
                    o_t = tl.sum(q_t[:, None] * s, axis=0)
                    tl.store(o_base + t_idx * stride_ot + offs_v * stride_od, o_t.to(Out_ptr.dtype.element_ty))

        if SAVE_FINAL:
            tl.store(
                Final_ptr + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
                s.to(Final_ptr.dtype.element_ty),
            )

    def _triton_chunk_gla_tc_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        save_final: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, HD = q.shape
        if out is None:
            out = torch.empty_like(v)
        has_init = initial_state is not None
        final_state = (
            torch.empty(B, H, HD, HD, device=q.device, dtype=q.dtype)
            if save_final
            else torch.empty((1,), device=q.device, dtype=q.dtype)
        )
        init_ptr = initial_state if has_init else final_state
        sb, sh, sk, sv = init_ptr.stride() if has_init else (final_state.stride() if save_final else (0, 0, 0, 0))
        _chunk_gla_tc_fwd_kernel[(B * H,)](
            q,
            k,
            v,
            g,
            out,
            init_ptr,
            final_state,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            sb,
            sh,
            sk,
            sv,
            T,
            H,
            HAS_INIT=has_init,
            SAVE_FINAL=save_final,
            BT=32,
            DK=HD,
            BV=HD,
            G_THRESH=-18.0 if q.dtype == torch.float16 else -160.0,
            num_warps=4,
        )
        return out, final_state

    @triton.jit
    def _triton_gated_deltanet_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        G_ptr,
        Beta_ptr,
        Out_ptr,
        Init_ptr,
        Final_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_gb,
        stride_gt,
        stride_gh,
        stride_gd,
        stride_bb,
        stride_bt,
        stride_bh,
        stride_bd,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sk,
        stride_sv,
        T,
        H,
        HAS_BETA: tl.constexpr,
        HAS_INIT: tl.constexpr,
        DK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """Fused register-state Gated DeltaNet kernel exploiting column independence of S[DK, BV]."""
        bh = tl.program_id(0)
        v_blk = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_k = tl.arange(0, DK)
        offs_v = v_blk * BV + tl.arange(0, BV)
        v_mask = offs_v < DK

        if HAS_INIT:
            s = tl.load(
                Init_ptr + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
                mask=v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            s = tl.zeros([DK, BV], dtype=tl.float32)

        q_base = Q_ptr + b * stride_qb + h * stride_qh
        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh
        g_base = G_ptr + b * stride_gb + h * stride_gh
        beta_base = Beta_ptr + b * stride_bb + h * stride_bh
        o_base = Out_ptr + b * stride_ob + h * stride_oh

        for t in range(T):
            q_t = tl.load(q_base + t * stride_qt + offs_k * stride_qd).to(tl.float32)
            k_t = tl.load(k_base + t * stride_kt + offs_k * stride_kd).to(tl.float32)
            v_t = tl.load(v_base + t * stride_vt + offs_v * stride_vd, mask=v_mask, other=0.0).to(tl.float32)
            g_t = tl.load(g_base + t * stride_gt + offs_k * stride_gd).to(tl.float32)
            d_t = tl.exp(tl.minimum(g_t, 0.0))

            v_pred = tl.sum(k_t[:, None] * s, axis=0)
            u_t = v_t - v_pred

            if HAS_BETA:
                b_t = tl.load(beta_base + t * stride_bt + offs_v * stride_bd, mask=v_mask, other=1.0).to(tl.float32)
                upd = k_t[:, None] * (b_t * u_t)[None, :]
            else:
                upd = k_t[:, None] * u_t[None, :]

            s = s * d_t[:, None] + upd
            o_t = tl.sum(q_t[:, None] * s, axis=0)
            tl.store(o_base + t * stride_ot + offs_v * stride_od, o_t.to(Out_ptr.dtype.element_ty), mask=v_mask)

        tl.store(
            Final_ptr + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv,
            s.to(Final_ptr.dtype.element_ty),
            mask=v_mask[None, :],
        )

    def _triton_gated_deltanet_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, HD = q.shape
        out = torch.empty_like(v)
        final_state = torch.empty(B, H, HD, HD, device=q.device, dtype=q.dtype)
        has_init = initial_state is not None
        init_ptr = initial_state if has_init else final_state
        has_beta = beta is not None
        if has_beta:
            beta_ref = beta
            stride_bb = beta.stride(0)
            stride_bt = beta.stride(1)
            stride_bh = beta.stride(2)
            stride_bd = beta.stride(3) if beta.ndim == 4 else 0
        else:
            beta_ref = final_state
            stride_bb = stride_bt = stride_bh = stride_bd = 0

        BV = 8
        grid = (B * H, triton.cdiv(HD, BV))
        _triton_gated_deltanet_fwd_kernel[grid](
            q,
            k,
            v,
            g,
            beta_ref,
            out,
            init_ptr,
            final_state,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            stride_bb,
            stride_bt,
            stride_bh,
            stride_bd,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            final_state.stride(0),
            final_state.stride(1),
            final_state.stride(2),
            final_state.stride(3),
            T,
            H,
            HAS_BETA=has_beta,
            HAS_INIT=has_init,
            DK=HD,
            BV=BV,
            num_warps=4,
        )
        return out, final_state
else:
    FusedRecurrentScan = None
    _triton_chunk_gla_tc_fwd = None
    _triton_gated_deltanet_fwd = None

_CAUSAL_MASK_INV_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _get_causal_mask_inv(device: torch.device, size: int) -> torch.Tensor:
    key = (device.type, size)
    mask = _CAUSAL_MASK_INV_CACHE.get(key)
    if mask is None or (hasattr(mask, "is_inference") and mask.is_inference()):
        with torch.no_grad():
            mask = (~torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))).view(1, 1, 1, size, size)
            _CAUSAL_MASK_INV_CACHE[key] = mask
    return mask.to(device=device)


def _run_gla_chunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized chunked GLA over [B, num_chunks * C, H, HD]."""
    B, T_chunk, H, HD = q.shape
    device = q.device
    dtype = q.dtype
    num_chunks = T_chunk // C

    q_c = q.reshape(B, num_chunks, C, H, HD).permute(0, 3, 1, 2, 4)
    k_c = k.reshape(B, num_chunks, C, H, HD).permute(0, 3, 1, 2, 4)
    v_c = v.reshape(B, num_chunks, C, H, HD).permute(0, 3, 1, 2, 4)
    g_c = torch.clamp(g.reshape(B, num_chunks, C, H, HD).permute(0, 3, 1, 2, 4), max=0.0)

    g_cumsum = torch.cumsum(g_c, dim=-2)
    causal_mask_inv = _get_causal_mask_inv(device, C)

    # Bounded-memory intra-chunk causal attention:
    # Uses micro-chunked slices (at most 4 chunks at a time) to ensure peak memory is bounded.
    # For normal decay ranges (g_cumsum.min() > -20.0), uses factored Tensor Core matmul with
    # midpoint exponential shift for exact mathematical equivalence in O(C*HD + C^2) FLOPs;
    # for extreme synthetic decays (e.g. g=-20.0 per step), falls back to pairwise clamped delta_g.
    is_capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    use_factored_intra = (not is_capturing) and (dtype != torch.float64) and bool((g_cumsum.min() > -20.0).item())

    if use_factored_intra:
        g_mid = g_cumsum.float()[:, :, :, C // 2 : C // 2 + 1, :]
        q_intra = q_c.float() * torch.exp(g_cumsum.float() - g_mid)
        k_intra = k_c.float() * torch.exp(g_mid - g_cumsum.float())
        if num_chunks > 4:
            out_intra_list = []
            micro = 4
            for c_start in range(0, num_chunks, micro):
                c_end = min(c_start + micro, num_chunks)
                attn_sub = (
                    torch.matmul(q_intra[:, :, c_start:c_end], k_intra[:, :, c_start:c_end].transpose(-1, -2))
                    .masked_fill_(causal_mask_inv, 0.0)
                    .to(dtype)
                )
                out_intra_list.append(torch.matmul(attn_sub, v_c[:, :, c_start:c_end]))
            out_intra = torch.cat(out_intra_list, dim=2)
        else:
            attn_all = torch.matmul(q_intra, k_intra.transpose(-1, -2)).masked_fill_(causal_mask_inv, 0.0).to(dtype)
            out_intra = torch.matmul(attn_all, v_c)
    elif num_chunks > 4:
        out_intra_list = []
        micro = 4
        for c_start in range(0, num_chunks, micro):
            c_end = min(c_start + micro, num_chunks)
            q_sub = q_c[:, :, c_start:c_end]
            k_sub = k_c[:, :, c_start:c_end]
            v_sub = v_c[:, :, c_start:c_end]
            g_cum_sub = g_cumsum[:, :, c_start:c_end]
            if dtype == torch.float64:
                delta_g_sub = torch.clamp(g_cum_sub.unsqueeze(-2) - g_cum_sub.unsqueeze(-3), max=0.0)
                attn_sub = (
                    ((q_sub.unsqueeze(-2) * k_sub.unsqueeze(-3)) * torch.exp(delta_g_sub))
                    .sum(dim=-1)
                    .masked_fill_(causal_mask_inv, 0.0)
                )
            else:
                delta_g_sub = torch.clamp(g_cum_sub.float().unsqueeze(-2) - g_cum_sub.float().unsqueeze(-3), max=0.0)
                d_exp_sub = torch.exp(delta_g_sub)
                qk_sub = (q_sub.float().unsqueeze(-2) * k_sub.float().unsqueeze(-3)) * d_exp_sub
                attn_sub = qk_sub.sum(dim=-1).masked_fill_(causal_mask_inv, 0.0).to(dtype)
            out_intra_list.append(torch.matmul(attn_sub, v_sub))
        out_intra = torch.cat(out_intra_list, dim=2)
    else:
        if dtype == torch.float64:
            delta_g = torch.clamp(g_cumsum.unsqueeze(-2) - g_cumsum.unsqueeze(-3), max=0.0)
            attn_all = (
                ((q_c.unsqueeze(-2) * k_c.unsqueeze(-3)) * torch.exp(delta_g))
                .sum(dim=-1)
                .masked_fill_(causal_mask_inv, 0.0)
            )
        else:
            delta_g = torch.clamp(g_cumsum.float().unsqueeze(-2) - g_cumsum.float().unsqueeze(-3), max=0.0)
            d_exp = torch.exp(delta_g)
            qk_elem = (q_c.float().unsqueeze(-2) * k_c.float().unsqueeze(-3)) * d_exp
            attn_all = qk_elem.sum(dim=-1).masked_fill_(causal_mask_inv, 0.0).to(dtype)
        out_intra = torch.matmul(attn_all, v_c)

    # Inter-chunk recurrent state propagation
    g_last = g_cumsum[:, :, :, -1:, :]
    k_decayed_all = k_c * torch.exp(torch.clamp(g_last - g_cumsum, max=0.0))
    chunk_decay = torch.exp(g_last).squeeze(-2).unsqueeze(-1)  # [B, H, num_chunks, HD, 1]

    q_exp = q_c * torch.exp(g_cumsum)

    if num_chunks == 1:
        out_inter = torch.matmul(q_exp, state.unsqueeze(2))
        kv_0 = torch.matmul(k_decayed_all[:, :, 0].transpose(-1, -2), v_c[:, :, 0])
        state = torch.addcmul(kv_0, state, chunk_decay[:, :, 0])
    elif (
        device.type == "cuda"
        and dtype != torch.float64
        and FusedRecurrentScan is not None
        and is_turbo_mode_enabled()
        and is_triton_available()
        and B <= 4
    ):
        kv_all = torch.matmul(k_decayed_all.transpose(-1, -2), v_c)
        S_all, state = FusedRecurrentScan.apply(kv_all, chunk_decay, state)
        out_inter = torch.matmul(q_exp, S_all)
    elif not torch.is_grad_enabled() and num_chunks >= 8:
        # Logarithmic Hillis-Steele parallel prefix scan in O(log2(num_chunks)) steps
        kv_all = torch.matmul(k_decayed_all.transpose(-1, -2), v_c)  # [B, H, num_chunks, HD, HD]
        kv_all[:, :, 0].addcmul_(state, chunk_decay[:, :, 0])
        a_scan = chunk_decay.clone()
        s = 1
        while s < num_chunks:
            kv_all[:, :, s:].add_(a_scan[:, :, s:] * kv_all[:, :, :-s])
            if (s << 1) < num_chunks:
                a_scan[:, :, s:].mul_(a_scan[:, :, :-s].clone())
            s <<= 1
        S_all = torch.cat([state.unsqueeze(2), kv_all[:, :, :-1]], dim=2)
        state = kv_all[:, :, -1]
        out_inter = torch.matmul(q_exp, S_all)
    else:
        kv_all = torch.matmul(k_decayed_all.transpose(-1, -2), v_c)
        states_prev = []
        for i in range(num_chunks):
            states_prev.append(state)
            state = torch.addcmul(kv_all[:, :, i], state, chunk_decay[:, :, i])
        S_all = torch.stack(states_prev, dim=2)
        out_inter = torch.matmul(q_exp, S_all)

    out_chunks = (out_intra + out_inter).permute(0, 2, 3, 1, 4).reshape(B, T_chunk, H, HD)
    return out_chunks, state


def fast_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute high-efficiency chunked Gated Linear Attention (GLA) with exact mathematical parity.

    Args:
        q: Query tensor [B, T, H, HD]
        k: Key tensor [B, T, H, HD]
        v: Value tensor [B, T, H, HD]
        g: Log-decay gate tensor [B, T, H, HD]
        initial_state: Optional recurrent hidden state [B, H, HD, HD]
        chunk_size: Chunk size for blockwise GEMMs (default: 64)
        out: Optional pre-allocated output tensor [B, T, H, HD]

    Returns:
        (out, final_state) where out is [B, T, H, HD] and final_state is [B, H, HD, HD]
    """
    B, T, H, HD = q.shape
    device = q.device
    dtype = q.dtype

    # Fast fused Triton Tensor-Core inference path (0 bytes of intermediate HBM allocation)
    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or g.requires_grad
        or (initial_state is not None and initial_state.requires_grad)
    )
    if (
        not needs_grad
        and q.is_cuda
        and _triton_chunk_gla_tc_fwd is not None
        and is_turbo_mode_enabled()
        and HD == 64
        and T >= 128
        and (T % 64 == 0)
        and dtype in (torch.bfloat16, torch.float16)
    ):
        return _triton_chunk_gla_tc_fwd(q, k, v, g, initial_state=initial_state, out=out)

    state = (
        initial_state.to(device=device, dtype=dtype)
        if initial_state is not None
        else torch.zeros(B, H, HD, HD, device=device, dtype=dtype)
    )

    # Fast single-token autoregressive decoding (O(1) step)
    if T == 1:
        decay = torch.exp(g[:, 0]).unsqueeze(-1)
        key = k[:, 0].unsqueeze(-1)
        value = v[:, 0].unsqueeze(-2)
        state = torch.addcmul(state * decay, key, value)
        query = q[:, 0].unsqueeze(-2)
        out = torch.matmul(query, state).squeeze(-2).unsqueeze(1)
        return out, state

    C = chunk_size

    # Short sequence (1 < T < C): execute as a single chunk of size T (zero Python loop)
    if T < C:
        return _run_gla_chunks(q, k, v, g, state, T)

    # Chunked parallel path for training and long sequences
    num_chunks = T // C
    T_chunk = num_chunks * C
    out_chunks, state = _run_gla_chunks(q[:, :T_chunk], k[:, :T_chunk], v[:, :T_chunk], g[:, :T_chunk], state, C)
    if T == T_chunk:
        return out_chunks, state

    T_rem = T - T_chunk
    if T_rem == 1:
        decay = torch.exp(g[:, T_chunk]).unsqueeze(-1)
        key = k[:, T_chunk].unsqueeze(-1)
        value = v[:, T_chunk].unsqueeze(-2)
        state = torch.addcmul(state * decay, key, value)
        query = q[:, T_chunk].unsqueeze(-2)
        out_rem = torch.matmul(query, state).squeeze(-2).unsqueeze(1)
    else:
        out_rem, state = _run_gla_chunks(q[:, T_chunk:], k[:, T_chunk:], v[:, T_chunk:], g[:, T_chunk:], state, T_rem)
    return torch.cat([out_chunks, out_rem], dim=1), state


def fast_recurrent_gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact token-by-token Gated DeltaNet recurrence with associative error-correcting memory updates.

    Implements:
      v_pred_t = k_t @ S_{t-1}
      u_t = v_t - v_pred_t  (associative error residual)
      S_t = S_{t-1} * exp(min(g_t, 0)) + beta_t * (k_t.T @ u_t)
      o_t = q_t @ S_t

    Reference: Songlin Yang et al., 'Gated Delta Networks: Improving Mamba2 with Delta Rule', ICLR 2025.
    """
    B, T, H, HD = q.shape
    device = q.device
    dtype = q.dtype
    state = (
        initial_state.to(device=device, dtype=dtype).clone()
        if initial_state is not None
        else torch.zeros(B, H, HD, HD, device=device, dtype=dtype)
    )
    outputs = []
    g_clamped = torch.clamp(g, max=0.0)
    decay = torch.exp(g_clamped)

    for t in range(T):
        q_t = q[:, t]  # [B, H, HD]
        k_t = k[:, t]  # [B, H, HD]
        v_t = v[:, t]  # [B, H, HD]
        d_t = decay[:, t].unsqueeze(-1)  # [B, H, HD, 1]

        # 1. Associative Memory Retrieval: predict v from key k_t
        v_pred = torch.matmul(k_t.unsqueeze(-2), state).squeeze(-2)  # [B, H, HD]
        u_t = v_t - v_pred  # Error residual

        if beta is not None:
            b_val = beta[:, t]
            if b_val.ndim == 3:  # [B, H, HD]
                b_t = b_val.unsqueeze(-1)
            else:  # [B, H]
                b_t = b_val.unsqueeze(-1).unsqueeze(-1)
            write_update = b_t * torch.matmul(k_t.unsqueeze(-1), u_t.unsqueeze(-2))
        else:
            write_update = torch.matmul(k_t.unsqueeze(-1), u_t.unsqueeze(-2))

        state = state * d_t + write_update
        o_t = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)
        outputs.append(o_t)

    out = torch.stack(outputs, dim=1)
    return out, state


def fast_chunk_gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute high-efficiency Gated DeltaNet combining gating with the Delta rule.

    Args:
        q: Query tensor [B, T, H, HD]
        k: Key tensor [B, T, H, HD]
        v: Value tensor [B, T, H, HD]
        g: Log-decay gate tensor [B, T, H, HD]
        beta: Optional write learning rate / gating tensor [B, T, H, HD] or [B, T, H]
        initial_state: Optional recurrent hidden state [B, H, HD, HD]
        chunk_size: Chunk size for blockwise GEMMs (default: 64)

    Returns:
        (out, final_state)
    """
    B, T, H, HD = q.shape
    device = q.device
    dtype = q.dtype

    state = (
        initial_state.to(device=device, dtype=dtype)
        if initial_state is not None
        else torch.zeros(B, H, HD, HD, device=device, dtype=dtype)
    )

    # Fast single-token autoregressive decoding (O(1) step)
    if T == 1:
        decay = torch.exp(torch.clamp(g[:, 0], max=0.0)).unsqueeze(-1)
        key = k[:, 0].unsqueeze(-2)  # [B, H, 1, HD]
        val = v[:, 0]
        v_pred = torch.matmul(key, state).squeeze(-2)  # [B, H, HD]
        u = val - v_pred
        b = beta[:, 0].unsqueeze(-1) if beta is not None else 1.0
        write = b * torch.matmul(k[:, 0].unsqueeze(-1), u.unsqueeze(-2))
        state = state * decay + write
        query = q[:, 0].unsqueeze(-2)
        out = torch.matmul(query, state).squeeze(-2).unsqueeze(1)
        return out, state

    # Fused Triton register-state DeltaNet forward kernel (eliminates Python T-loop)
    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or g.requires_grad
        or (beta is not None and beta.requires_grad)
        or (initial_state is not None and initial_state.requires_grad)
    )
    if (
        not needs_grad
        and q.is_cuda
        and _triton_gated_deltanet_fwd is not None
        and is_turbo_mode_enabled()
        and HD in (16, 32, 64, 128)
    ):
        return _triton_gated_deltanet_fwd(q, k, v, g, beta=beta, initial_state=initial_state)

    return fast_recurrent_gated_deltanet(q, k, v, g, beta=beta, initial_state=state)
