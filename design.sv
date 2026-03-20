`timescale 1ns/1ps

// MAC Unit — Multiply-Accumulate with kernel stride
// Computes: result = Σ in[i] * w[i]  for i = 0..KERNEL_SIZE*KERNEL_SIZE-1
//
// Diagram: in[i] → |MAC kernel stride| → Σ i·w
//                  w[i]  →

module mac_unit #(
    parameter DATA_WIDTH    = 8,              // bit-width of inputs/weights
    parameter KERNEL_SIZE   = 5,              // e.g. 5x5 LeNet kernel
    parameter NUM_ELEMENTS  = KERNEL_SIZE * KERNEL_SIZE,  // total elements to accumulate
    parameter ACC_WIDTH     = DATA_WIDTH * 2 + $clog2(NUM_ELEMENTS)
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           en,         // pulse high to start
    // flattened kernel window: NUM_ELEMENTS elements
    input  logic signed [DATA_WIDTH-1:0]   in_data [0:NUM_ELEMENTS-1],
    input  logic signed [DATA_WIDTH-1:0]   weight  [0:NUM_ELEMENTS-1],
    output logic signed [ACC_WIDTH-1:0]    result,
    output logic                           valid       // high when result is ready
);

    logic signed [ACC_WIDTH-1:0] acc;
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            valid <= 1'b0;
        end else if (en) begin
            acc = '0;
            // Accumulate in[i] * w[i] for each element in the kernel window
            for (i = 0; i < NUM_ELEMENTS; i++) begin
                acc = acc + (in_data[i] * weight[i]);
            end
            result <= acc;
            valid  <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule

// Convolution Module — computes all output positions in one cycle
// using inline dot products (no MAC submodule needed).

module conv_hw #(
    parameter DATA_WIDTH   = 8,
    parameter KERNEL_SIZE  = 5,          // LeNet uses 5x5 kernels
    parameter IMG_H        = 32,
    parameter IMG_W        = 32,
    parameter IN_CH        = 1,          // input channels
    parameter OUT_H        = IMG_H - KERNEL_SIZE + 1,   // valid convolution output height
    parameter OUT_W        = IMG_W - KERNEL_SIZE + 1,
    parameter ACC_WIDTH    = DATA_WIDTH * 2 + $clog2(KERNEL_SIZE * KERNEL_SIZE * IN_CH)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    // Input feature map: [channel][row][col]
    input  logic signed [DATA_WIDTH-1:0]  input_fm [0:IN_CH-1][0:IMG_H-1][0:IMG_W-1],

    // Convolution kernel weights: [in_channel][kernel_row][kernel_col]
    input  logic signed [DATA_WIDTH-1:0]  weight   [0:IN_CH-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],

    // Output feature map result: [out_row][out_col]
    output logic signed [ACC_WIDTH-1:0]   result   [0:OUT_H-1][0:OUT_W-1],
    output logic                          ready     // high when entire output map is computed
);

    typedef enum logic [1:0] { IDLE, DONE } state_t;
    state_t state;

    logic signed [ACC_WIDTH-1:0] acc;
    integer orow, ocol, ch, kr, kc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            ready <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    ready <= 1'b0;
                    if (start) begin
                        for (orow = 0; orow < OUT_H; orow++) begin
                            for (ocol = 0; ocol < OUT_W; ocol++) begin
                                acc = '0;
                                for (ch = 0; ch < IN_CH; ch++)
                                    for (kr = 0; kr < KERNEL_SIZE; kr++)
                                        for (kc = 0; kc < KERNEL_SIZE; kc++)
                                            acc = acc + (input_fm[ch][orow + kr][ocol + kc] * weight[ch][kr][kc]);
                                result[orow][ocol] = acc;
                            end
                        end
                        state <= DONE;
                    end
                end

                DONE: begin
                    ready <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

// Pooling Module — Average and Max pooling
// Computes all output positions in one cycle.

module pool #(
    parameter DATA_WIDTH  = 8,
    parameter POOL_SIZE   = 2,           // 2x2 window (standard for LeNet)
    parameter IN_H        = 28,          // input feature map height
    parameter IN_W        = 28,          // input feature map width
    parameter OUT_H       = IN_H / POOL_SIZE,
    parameter OUT_W       = IN_W / POOL_SIZE,
    parameter MODE        = 0            // 0 = AVG (LeNet default), 1 = MAX
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    // Input feature map
    input  logic signed [DATA_WIDTH-1:0]  i [0:IN_H-1][0:IN_W-1],

    // Pooled output
    output logic signed [DATA_WIDTH-1:0]  result [0:OUT_H-1][0:OUT_W-1],
    output logic                          valid   // high for one cycle when output is ready
);

    localparam NUM_POOL = POOL_SIZE * POOL_SIZE;
    localparam SUM_WIDTH = DATA_WIDTH + $clog2(NUM_POOL);

    typedef enum logic [1:0] { IDLE, DONE } state_t;
    state_t state;

    integer pr, pc, wr, wc;
    logic signed [SUM_WIDTH-1:0] sum;
    logic signed [DATA_WIDTH-1:0] max_val;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    valid <= 1'b0;
                    if (start) begin
                        for (pr = 0; pr < OUT_H; pr++) begin
                            for (pc = 0; pc < OUT_W; pc++) begin
                                sum = '0;
                                max_val = i[pr * POOL_SIZE][pc * POOL_SIZE];
                                for (wr = 0; wr < POOL_SIZE; wr++) begin
                                    for (wc = 0; wc < POOL_SIZE; wc++) begin
                                        sum = sum + i[pr * POOL_SIZE + wr][pc * POOL_SIZE + wc];
                                        if (i[pr * POOL_SIZE + wr][pc * POOL_SIZE + wc] > max_val)
                                            max_val = i[pr * POOL_SIZE + wr][pc * POOL_SIZE + wc];
                                    end
                                end
                                if (MODE == 0)
                                    result[pr][pc] = sum >>> $clog2(NUM_POOL);
                                else
                                    result[pr][pc] = max_val;
                            end
                        end
                        state <= DONE;
                    end
                end

                DONE: begin
                    valid <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

// ReLU Activation Module — Element-wise ReLU
// Processes all elements in one cycle.

module relu #(
    parameter DATA_WIDTH = 8,
    parameter NUM_ELEMENTS = 120       // number of elements to activate
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    input  logic signed [DATA_WIDTH-1:0]  in_data  [0:NUM_ELEMENTS-1],

    output logic signed [DATA_WIDTH-1:0]  out_data [0:NUM_ELEMENTS-1],
    output logic                          valid
);

    typedef enum logic [1:0] { IDLE, DONE } state_t;
    state_t state;

    integer j;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    valid <= 1'b0;
                    if (start) begin
                        for (j = 0; j < NUM_ELEMENTS; j++) begin
                            if (in_data[j] < 0)
                                out_data[j] = '0;
                            else
                                out_data[j] = in_data[j];
                        end
                        state <= DONE;
                    end
                end

                DONE: begin
                    valid <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

// Dense (Fully Connected) Layer Module
// Computes all output neurons in one cycle.

module dense #(
    parameter DATA_WIDTH    = 8,
    parameter IN_FEATURES   = 120,       // number of input neurons
    parameter OUT_FEATURES  = 84,        // number of output neurons
    parameter ACC_WIDTH     = DATA_WIDTH * 2 + $clog2(IN_FEATURES)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    // Input vector
    input  logic signed [DATA_WIDTH-1:0]  in_data [0:IN_FEATURES-1],

    // Weight matrix: [out_neuron][in_neuron]
    input  logic signed [DATA_WIDTH-1:0]  weight  [0:OUT_FEATURES-1][0:IN_FEATURES-1],

    // Bias vector
    input  logic signed [ACC_WIDTH-1:0]   bias    [0:OUT_FEATURES-1],

    // Output vector
    output logic signed [ACC_WIDTH-1:0]   out_data [0:OUT_FEATURES-1],
    output logic                          valid
);

    typedef enum logic [1:0] { IDLE, DONE } state_t;
    state_t state;

    logic signed [ACC_WIDTH-1:0] acc;
    integer j, k;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    valid <= 1'b0;
                    if (start) begin
                        for (j = 0; j < OUT_FEATURES; j++) begin
                            acc = bias[j];
                            for (k = 0; k < IN_FEATURES; k++)
                                acc = acc + (in_data[k] * weight[j][k]);
                            out_data[j] = acc;
                        end
                        state <= DONE;
                    end
                end

                DONE: begin
                    valid <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

// ============================================================================
// LeNet-5 Top-Level Module
// ============================================================================
//
// Architecture:
//   Input(32x32x1) -> C1(6@28x28) -> ReLU -> S2(6@14x14)
//                  -> C3(16@10x10) -> ReLU -> S4(16@5x5)
//                  -> Flatten(400) -> FC5(120) -> ReLU
//                  -> FC6(84) -> ReLU -> FC7(10) -> Output
//
// ============================================================================

module lenet #(
    parameter DATA_WIDTH  = 8,
    // Derived accumulator widths (do not override)
    parameter C1_ACC_W    = DATA_WIDTH * 2 + $clog2(1 * 5 * 5),
    parameter C3_ACC_W    = DATA_WIDTH * 2 + $clog2(6 * 5 * 5),
    parameter FC5_ACC_W   = DATA_WIDTH * 2 + $clog2(400),
    parameter FC6_ACC_W   = DATA_WIDTH * 2 + $clog2(120),
    parameter FC7_ACC_W   = DATA_WIDTH * 2 + $clog2(84)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    // Input image (32x32 grayscale)
    input  logic signed [DATA_WIDTH-1:0]  image [0:31][0:31],

    // C1 weights: 6 filters x (1 input channel x 5x5 kernel)
    input  logic signed [DATA_WIDTH-1:0]  c1_weight [0:5][0:0][0:4][0:4],

    // C3 weights: 16 filters x (6 input channels x 5x5 kernel)
    input  logic signed [DATA_WIDTH-1:0]  c3_weight [0:15][0:5][0:4][0:4],

    // FC5: 400 -> 120
    input  logic signed [DATA_WIDTH-1:0]  fc5_weight [0:119][0:399],
    input  logic signed [FC5_ACC_W-1:0]   fc5_bias   [0:119],

    // FC6: 120 -> 84
    input  logic signed [DATA_WIDTH-1:0]  fc6_weight [0:83][0:119],
    input  logic signed [FC6_ACC_W-1:0]   fc6_bias   [0:83],

    // FC7: 84 -> 10
    input  logic signed [DATA_WIDTH-1:0]  fc7_weight [0:9][0:83],
    input  logic signed [FC7_ACC_W-1:0]   fc7_bias   [0:9],

    // Output: 10 class scores (raw logits, no activation on final layer)
    output logic signed [FC7_ACC_W-1:0]   scores [0:9],
    output logic                          done
);

    // =====================================================================
    //  State machine — sequences through all LeNet layers
    // =====================================================================
    typedef enum logic [3:0] {
        IDLE, L_C1, L_S2, L_C3, L_S4, L_FC5, L_FC6, L_FC7, L_DONE
    } state_t;
    state_t state;
    logic   layer_started;

    // Start pulses: high for exactly one cycle when entering each layer
    wire c1_go  = (state == L_C1)  && !layer_started;
    wire s2_go  = (state == L_S2)  && !layer_started;
    wire c3_go  = (state == L_C3)  && !layer_started;
    wire s4_go  = (state == L_S4)  && !layer_started;
    wire fc5_go = (state == L_FC5) && !layer_started;
    wire fc6_go = (state == L_FC6) && !layer_started;
    wire fc7_go = (state == L_FC7) && !layer_started;

    // Maximum positive value for DATA_WIDTH-bit signed (e.g. 127 for 8-bit)
    localparam MAX_POS = (1 << (DATA_WIDTH - 1)) - 1;

    // Done flags (packed for reduction AND)
    logic [5:0]  c1_done_vec;
    logic [5:0]  s2_done_vec;
    logic [15:0] c3_done_vec;
    logic [15:0] s4_done_vec;
    logic        fc5_done, fc6_done, fc7_done;

    wire c1_all_done = &c1_done_vec;
    wire s2_all_done = &s2_done_vec;
    wire c3_all_done = &c3_done_vec;
    wire s4_all_done = &s4_done_vec;

    // =====================================================================
    //  C1: 6 x conv(1 ch, 5x5, 32x32 -> 28x28) + saturating ReLU
    // =====================================================================

    // Wrap 2D image as 3D [1 ch][32][32] for conv_hw
    logic signed [DATA_WIDTH-1:0] c1_fm [0:0][0:31][0:31];
    always_comb
        for (int r = 0; r < 32; r++)
            for (int c = 0; c < 32; c++)
                c1_fm[0][r][c] = image[r][c];

    // Activated output: truncated to DATA_WIDTH with ReLU
    logic signed [DATA_WIDTH-1:0] c1_act [0:5][0:27][0:27];

    genvar gf;
    generate
        for (gf = 0; gf < 6; gf++) begin : gen_c1
            logic signed [C1_ACC_W-1:0] raw [0:27][0:27];
            logic signed [DATA_WIDTH-1:0] local_wt [0:0][0:4][0:4];
            logic rdy;

            always_comb
                for (int kr = 0; kr < 5; kr++)
                    for (int kc = 0; kc < 5; kc++)
                        local_wt[0][kr][kc] = c1_weight[gf][0][kr][kc];

            conv_hw #(
                .DATA_WIDTH (DATA_WIDTH),
                .KERNEL_SIZE(5),
                .IMG_H      (32),
                .IMG_W      (32),
                .IN_CH      (1)
            ) u_conv (
                .clk      (clk),
                .rst_n    (rst_n),
                .start    (c1_go),
                .input_fm (c1_fm),
                .weight   (local_wt),
                .result   (raw),
                .ready    (rdy)
            );

            assign c1_done_vec[gf] = rdy;

            // Saturating ReLU: C1_ACC_W -> DATA_WIDTH
            always_comb
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                        if (raw[r][c] <= 0)
                            c1_act[gf][r][c] = '0;
                        else if (raw[r][c] > MAX_POS)
                            c1_act[gf][r][c] = MAX_POS;
                        else
                            c1_act[gf][r][c] = raw[r][c];
        end
    endgenerate

    // =====================================================================
    //  S2: 6 x avg_pool(28x28 -> 14x14)
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] s2_out [0:5][0:13][0:13];

    generate
        for (gf = 0; gf < 6; gf++) begin : gen_s2
            logic signed [DATA_WIDTH-1:0] pool_in  [0:27][0:27];
            logic signed [DATA_WIDTH-1:0] pool_out [0:13][0:13];
            logic rdy;

            always_comb
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                        pool_in[r][c] = c1_act[gf][r][c];

            pool #(
                .DATA_WIDTH(DATA_WIDTH),
                .IN_H      (28),
                .IN_W      (28),
                .POOL_SIZE (2),
                .MODE      (0)
            ) u_pool (
                .clk    (clk),
                .rst_n  (rst_n),
                .start  (s2_go),
                .i      (pool_in),
                .result (pool_out),
                .valid  (rdy)
            );

            assign s2_done_vec[gf] = rdy;

            always_comb
                for (int r = 0; r < 14; r++)
                    for (int c = 0; c < 14; c++)
                        s2_out[gf][r][c] = pool_out[r][c];
        end
    endgenerate

    // =====================================================================
    //  C3: 16 x conv(6 ch, 5x5, 14x14 -> 10x10) + saturating ReLU
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] c3_act [0:15][0:9][0:9];

    generate
        for (gf = 0; gf < 16; gf++) begin : gen_c3
            logic signed [C3_ACC_W-1:0] raw [0:9][0:9];
            logic signed [DATA_WIDTH-1:0] local_wt [0:5][0:4][0:4];
            logic rdy;

            always_comb
                for (int ch = 0; ch < 6; ch++)
                    for (int kr = 0; kr < 5; kr++)
                        for (int kc = 0; kc < 5; kc++)
                            local_wt[ch][kr][kc] = c3_weight[gf][ch][kr][kc];

            conv_hw #(
                .DATA_WIDTH (DATA_WIDTH),
                .KERNEL_SIZE(5),
                .IMG_H      (14),
                .IMG_W      (14),
                .IN_CH      (6)
            ) u_conv (
                .clk      (clk),
                .rst_n    (rst_n),
                .start    (c3_go),
                .input_fm (s2_out),
                .weight   (local_wt),
                .result   (raw),
                .ready    (rdy)
            );

            assign c3_done_vec[gf] = rdy;

            // Saturating ReLU: C3_ACC_W -> DATA_WIDTH
            always_comb
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 10; c++)
                        if (raw[r][c] <= 0)
                            c3_act[gf][r][c] = '0;
                        else if (raw[r][c] > MAX_POS)
                            c3_act[gf][r][c] = MAX_POS;
                        else
                            c3_act[gf][r][c] = raw[r][c];
        end
    endgenerate

    // =====================================================================
    //  S4: 16 x avg_pool(10x10 -> 5x5)
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] s4_out [0:15][0:4][0:4];

    generate
        for (gf = 0; gf < 16; gf++) begin : gen_s4
            logic signed [DATA_WIDTH-1:0] pool_in  [0:9][0:9];
            logic signed [DATA_WIDTH-1:0] pool_out [0:4][0:4];
            logic rdy;

            always_comb
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 10; c++)
                        pool_in[r][c] = c3_act[gf][r][c];

            pool #(
                .DATA_WIDTH(DATA_WIDTH),
                .IN_H      (10),
                .IN_W      (10),
                .POOL_SIZE (2),
                .MODE      (0)
            ) u_pool (
                .clk    (clk),
                .rst_n  (rst_n),
                .start  (s4_go),
                .i      (pool_in),
                .result (pool_out),
                .valid  (rdy)
            );

            assign s4_done_vec[gf] = rdy;

            always_comb
                for (int r = 0; r < 5; r++)
                    for (int c = 0; c < 5; c++)
                        s4_out[gf][r][c] = pool_out[r][c];
        end
    endgenerate

    // =====================================================================
    //  Flatten: 16 x 5x5 -> 400-element vector
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] fc5_in [0:399];

    always_comb
        for (int ch = 0; ch < 16; ch++)
            for (int r = 0; r < 5; r++)
                for (int c = 0; c < 5; c++)
                    fc5_in[ch * 25 + r * 5 + c] = s4_out[ch][r][c];

    // =====================================================================
    //  FC5: dense(400 -> 120) + saturating ReLU
    // =====================================================================
    logic signed [FC5_ACC_W-1:0]  fc5_raw [0:119];
    logic signed [DATA_WIDTH-1:0] fc6_in  [0:119];

    dense #(
        .DATA_WIDTH  (DATA_WIDTH),
        .IN_FEATURES (400),
        .OUT_FEATURES(120)
    ) u_fc5 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (fc5_go),
        .in_data (fc5_in),
        .weight  (fc5_weight),
        .bias    (fc5_bias),
        .out_data(fc5_raw),
        .valid   (fc5_done)
    );

    always_comb
        for (int j = 0; j < 120; j++)
            if (fc5_raw[j] <= 0)
                fc6_in[j] = '0;
            else if (fc5_raw[j] > MAX_POS)
                fc6_in[j] = MAX_POS;
            else
                fc6_in[j] = fc5_raw[j];

    // =====================================================================
    //  FC6: dense(120 -> 84) + saturating ReLU
    // =====================================================================
    logic signed [FC6_ACC_W-1:0]  fc6_raw [0:83];
    logic signed [DATA_WIDTH-1:0] fc7_in  [0:83];

    dense #(
        .DATA_WIDTH  (DATA_WIDTH),
        .IN_FEATURES (120),
        .OUT_FEATURES(84)
    ) u_fc6 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (fc6_go),
        .in_data (fc6_in),
        .weight  (fc6_weight),
        .bias    (fc6_bias),
        .out_data(fc6_raw),
        .valid   (fc6_done)
    );

    always_comb
        for (int j = 0; j < 84; j++)
            if (fc6_raw[j] <= 0)
                fc7_in[j] = '0;
            else if (fc6_raw[j] > MAX_POS)
                fc7_in[j] = MAX_POS;
            else
                fc7_in[j] = fc6_raw[j];

    // =====================================================================
    //  FC7: dense(84 -> 10) — no ReLU on output (raw logits)
    // =====================================================================
    dense #(
        .DATA_WIDTH  (DATA_WIDTH),
        .IN_FEATURES (84),
        .OUT_FEATURES(10)
    ) u_fc7 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (fc7_go),
        .in_data (fc7_in),
        .weight  (fc7_weight),
        .bias    (fc7_bias),
        .out_data(scores),
        .valid   (fc7_done)
    );

    // =====================================================================
    //  Main sequencing state machine
    // =====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            layer_started <= 1'b0;
            done          <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done          <= 1'b0;
                    layer_started <= 1'b0;
                    if (start) state <= L_C1;
                end

                L_C1: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (c1_all_done) begin
                        state <= L_S2;  layer_started <= 1'b0;
                    end
                end

                L_S2: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (s2_all_done) begin
                        state <= L_C3;  layer_started <= 1'b0;
                    end
                end

                L_C3: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (c3_all_done) begin
                        state <= L_S4;  layer_started <= 1'b0;
                    end
                end

                L_S4: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (s4_all_done) begin
                        state <= L_FC5; layer_started <= 1'b0;
                    end
                end

                L_FC5: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc5_done) begin
                        state <= L_FC6; layer_started <= 1'b0;
                    end
                end

                L_FC6: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc6_done) begin
                        state <= L_FC7; layer_started <= 1'b0;
                    end
                end

                L_FC7: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc7_done) begin
                        state <= L_DONE; layer_started <= 1'b0;
                    end
                end

                L_DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
