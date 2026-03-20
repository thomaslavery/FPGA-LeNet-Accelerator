`timescale 1ns/1ps

// ============================================================================
// IVerilog workaround: all module OUTPUT arrays use flat packed vectors
// because IVerilog issue #1001 makes unpacked array output ports undefined.
//
// Access convention:  element [i] = flat_vec[i*WIDTH +: WIDTH]
//            2D [r][c] = flat_vec[(r*COLS+c)*WIDTH +: WIDTH]
// ============================================================================

// MAC Unit — scalar output, works as-is in IVerilog
module mac_unit #(
    parameter DATA_WIDTH    = 8,
    parameter KERNEL_SIZE   = 5,
    parameter NUM_ELEMENTS  = KERNEL_SIZE * KERNEL_SIZE,
    parameter ACC_WIDTH     = DATA_WIDTH * 2 + $clog2(NUM_ELEMENTS)
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           en,
    input  logic signed [DATA_WIDTH-1:0]   in_data [0:NUM_ELEMENTS-1],
    input  logic signed [DATA_WIDTH-1:0]   weight  [0:NUM_ELEMENTS-1],
    output logic signed [ACC_WIDTH-1:0]    result,
    output logic                           valid
);
    logic signed [ACC_WIDTH-1:0] acc;
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            valid <= 1'b0;
        end else if (en) begin
            acc = '0;
            for (i = 0; i < NUM_ELEMENTS; i++)
                acc = acc + (in_data[i] * weight[i]);
            result <= acc;
            valid  <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end
endmodule

// Convolution — all positions in one cycle
// Output: flat packed vector, element [r][c] at [(r*OUT_W+c)*ACC_WIDTH +: ACC_WIDTH]
module conv_hw #(
    parameter DATA_WIDTH   = 8,
    parameter KERNEL_SIZE  = 5,
    parameter IMG_H        = 32,
    parameter IMG_W        = 32,
    parameter IN_CH        = 1,
    parameter OUT_H        = IMG_H - KERNEL_SIZE + 1,
    parameter OUT_W        = IMG_W - KERNEL_SIZE + 1,
    parameter ACC_WIDTH    = DATA_WIDTH * 2 + $clog2(KERNEL_SIZE * KERNEL_SIZE * IN_CH)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_WIDTH-1:0]  input_fm [0:IN_CH-1][0:IMG_H-1][0:IMG_W-1],
    input  logic signed [DATA_WIDTH-1:0]  weight   [0:IN_CH-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    output logic [ACC_WIDTH*OUT_H*OUT_W-1:0] result_flat,
    output logic                          ready
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
                        for (orow = 0; orow < OUT_H; orow++)
                            for (ocol = 0; ocol < OUT_W; ocol++) begin
                                acc = '0;
                                for (ch = 0; ch < IN_CH; ch++)
                                    for (kr = 0; kr < KERNEL_SIZE; kr++)
                                        for (kc = 0; kc < KERNEL_SIZE; kc++)
                                            acc = acc + (input_fm[ch][orow+kr][ocol+kc] * weight[ch][kr][kc]);
                                result_flat[(orow*OUT_W+ocol)*ACC_WIDTH +: ACC_WIDTH] = acc;
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

// Pooling — all positions in one cycle
// Output: flat packed vector, element [r][c] at [(r*OUT_W+c)*DATA_WIDTH +: DATA_WIDTH]
module pool #(
    parameter DATA_WIDTH  = 8,
    parameter POOL_SIZE   = 2,
    parameter IN_H        = 28,
    parameter IN_W        = 28,
    parameter OUT_H       = IN_H / POOL_SIZE,
    parameter OUT_W       = IN_W / POOL_SIZE,
    parameter MODE        = 0
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_WIDTH-1:0]  i [0:IN_H-1][0:IN_W-1],
    output logic [DATA_WIDTH*OUT_H*OUT_W-1:0] result_flat,
    output logic                          valid
);
    localparam NUM_POOL  = POOL_SIZE * POOL_SIZE;
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
                        for (pr = 0; pr < OUT_H; pr++)
                            for (pc = 0; pc < OUT_W; pc++) begin
                                sum = '0;
                                max_val = i[pr*POOL_SIZE][pc*POOL_SIZE];
                                for (wr = 0; wr < POOL_SIZE; wr++)
                                    for (wc = 0; wc < POOL_SIZE; wc++) begin
                                        sum = sum + i[pr*POOL_SIZE+wr][pc*POOL_SIZE+wc];
                                        if (i[pr*POOL_SIZE+wr][pc*POOL_SIZE+wc] > max_val)
                                            max_val = i[pr*POOL_SIZE+wr][pc*POOL_SIZE+wc];
                                    end
                                if (MODE == 0)
                                    result_flat[(pr*OUT_W+pc)*DATA_WIDTH +: DATA_WIDTH] = sum >>> $clog2(NUM_POOL);
                                else
                                    result_flat[(pr*OUT_W+pc)*DATA_WIDTH +: DATA_WIDTH] = max_val;
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

// ReLU — all elements in one cycle
// Output: flat packed vector, element [i] at [i*DATA_WIDTH +: DATA_WIDTH]
module relu #(
    parameter DATA_WIDTH   = 8,
    parameter NUM_ELEMENTS = 120
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_WIDTH-1:0]  in_data [0:NUM_ELEMENTS-1],
    output logic [DATA_WIDTH*NUM_ELEMENTS-1:0] out_flat,
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
                                out_flat[j*DATA_WIDTH +: DATA_WIDTH] = '0;
                            else
                                out_flat[j*DATA_WIDTH +: DATA_WIDTH] = in_data[j];
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

// Dense (FC) — all neurons in one cycle
// Output: flat packed vector, element [j] at [j*ACC_WIDTH +: ACC_WIDTH]
module dense #(
    parameter DATA_WIDTH    = 8,
    parameter IN_FEATURES   = 120,
    parameter OUT_FEATURES  = 84,
    parameter ACC_WIDTH     = DATA_WIDTH * 2 + $clog2(IN_FEATURES)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_WIDTH-1:0]  in_data [0:IN_FEATURES-1],
    input  logic signed [DATA_WIDTH-1:0]  weight  [0:OUT_FEATURES-1][0:IN_FEATURES-1],
    input  logic signed [ACC_WIDTH-1:0]   bias    [0:OUT_FEATURES-1],
    output logic [ACC_WIDTH*OUT_FEATURES-1:0] out_flat,
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
                            out_flat[j*ACC_WIDTH +: ACC_WIDTH] = acc;
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
module lenet #(
    parameter DATA_WIDTH  = 8,
    parameter C1_ACC_W    = DATA_WIDTH * 2 + $clog2(1 * 5 * 5),
    parameter C3_ACC_W    = DATA_WIDTH * 2 + $clog2(6 * 5 * 5),
    parameter FC5_ACC_W   = DATA_WIDTH * 2 + $clog2(400),
    parameter FC6_ACC_W   = DATA_WIDTH * 2 + $clog2(120),
    parameter FC7_ACC_W   = DATA_WIDTH * 2 + $clog2(84)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,
    input  logic signed [DATA_WIDTH-1:0]  image      [0:31][0:31],
    input  logic signed [DATA_WIDTH-1:0]  c1_weight  [0:5][0:0][0:4][0:4],
    input  logic signed [DATA_WIDTH-1:0]  c3_weight  [0:15][0:5][0:4][0:4],
    input  logic signed [DATA_WIDTH-1:0]  fc5_weight [0:119][0:399],
    input  logic signed [FC5_ACC_W-1:0]   fc5_bias   [0:119],
    input  logic signed [DATA_WIDTH-1:0]  fc6_weight [0:83][0:119],
    input  logic signed [FC6_ACC_W-1:0]   fc6_bias   [0:83],
    input  logic signed [DATA_WIDTH-1:0]  fc7_weight [0:9][0:83],
    input  logic signed [FC7_ACC_W-1:0]   fc7_bias   [0:9],
    output logic [FC7_ACC_W*10-1:0]       scores_flat,
    output logic                          done
);

    // ---- State machine ----
    typedef enum logic [3:0] {
        IDLE, L_C1, L_S2, L_C3, L_S4, L_FC5, L_FC6, L_FC7, L_DONE
    } state_t;
    state_t state;
    logic   layer_started;

    wire c1_go  = (state == L_C1)  && !layer_started;
    wire s2_go  = (state == L_S2)  && !layer_started;
    wire c3_go  = (state == L_C3)  && !layer_started;
    wire s4_go  = (state == L_S4)  && !layer_started;
    wire fc5_go = (state == L_FC5) && !layer_started;
    wire fc6_go = (state == L_FC6) && !layer_started;
    wire fc7_go = (state == L_FC7) && !layer_started;

    localparam MAX_POS = (1 << (DATA_WIDTH - 1)) - 1;

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
    logic signed [DATA_WIDTH-1:0] c1_fm [0:0][0:31][0:31];
    always_comb
        for (int r = 0; r < 32; r++)
            for (int c = 0; c < 32; c++)
                c1_fm[0][r][c] = image[r][c];

    logic signed [DATA_WIDTH-1:0] c1_act [0:5][0:27][0:27];

    genvar gf;
    generate
        for (gf = 0; gf < 6; gf++) begin : gen_c1
            logic [C1_ACC_W*28*28-1:0] raw_flat;
            logic signed [DATA_WIDTH-1:0] local_wt [0:0][0:4][0:4];
            logic rdy;

            always_comb
                for (int kr = 0; kr < 5; kr++)
                    for (int kc = 0; kc < 5; kc++)
                        local_wt[0][kr][kc] = c1_weight[gf][0][kr][kc];

            conv_hw #(
                .DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(5),
                .IMG_H(32), .IMG_W(32), .IN_CH(1)
            ) u_conv (
                .clk(clk), .rst_n(rst_n), .start(c1_go),
                .input_fm(c1_fm), .weight(local_wt),
                .result_flat(raw_flat), .ready(rdy)
            );

            assign c1_done_vec[gf] = rdy;

            // Unpack + saturating ReLU
            always_comb
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++) begin
                        // Extract signed conv result from flat vector
                        if ($signed(raw_flat[(r*28+c)*C1_ACC_W +: C1_ACC_W]) <= 0)
                            c1_act[gf][r][c] = '0;
                        else if ($signed(raw_flat[(r*28+c)*C1_ACC_W +: C1_ACC_W]) > MAX_POS)
                            c1_act[gf][r][c] = MAX_POS;
                        else
                            c1_act[gf][r][c] = raw_flat[(r*28+c)*C1_ACC_W +: DATA_WIDTH];
                    end
        end
    endgenerate

    // =====================================================================
    //  S2: 6 x avg_pool(28x28 -> 14x14)
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] s2_out [0:5][0:13][0:13];

    generate
        for (gf = 0; gf < 6; gf++) begin : gen_s2
            logic signed [DATA_WIDTH-1:0] pool_in [0:27][0:27];
            logic [DATA_WIDTH*14*14-1:0] pool_flat;
            logic rdy;

            always_comb
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                        pool_in[r][c] = c1_act[gf][r][c];

            pool #(
                .DATA_WIDTH(DATA_WIDTH), .IN_H(28), .IN_W(28),
                .POOL_SIZE(2), .MODE(0)
            ) u_pool (
                .clk(clk), .rst_n(rst_n), .start(s2_go),
                .i(pool_in), .result_flat(pool_flat), .valid(rdy)
            );

            assign s2_done_vec[gf] = rdy;

            always_comb
                for (int r = 0; r < 14; r++)
                    for (int c = 0; c < 14; c++)
                        s2_out[gf][r][c] = $signed(pool_flat[(r*14+c)*DATA_WIDTH +: DATA_WIDTH]);
        end
    endgenerate

    // =====================================================================
    //  C3: 16 x conv(6 ch, 5x5, 14x14 -> 10x10) + saturating ReLU
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] c3_act [0:15][0:9][0:9];

    generate
        for (gf = 0; gf < 16; gf++) begin : gen_c3
            logic [C3_ACC_W*10*10-1:0] raw_flat;
            logic signed [DATA_WIDTH-1:0] local_wt [0:5][0:4][0:4];
            logic rdy;

            always_comb
                for (int ch = 0; ch < 6; ch++)
                    for (int kr = 0; kr < 5; kr++)
                        for (int kc = 0; kc < 5; kc++)
                            local_wt[ch][kr][kc] = c3_weight[gf][ch][kr][kc];

            conv_hw #(
                .DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(5),
                .IMG_H(14), .IMG_W(14), .IN_CH(6)
            ) u_conv (
                .clk(clk), .rst_n(rst_n), .start(c3_go),
                .input_fm(s2_out), .weight(local_wt),
                .result_flat(raw_flat), .ready(rdy)
            );

            assign c3_done_vec[gf] = rdy;

            always_comb
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 10; c++) begin
                        if ($signed(raw_flat[(r*10+c)*C3_ACC_W +: C3_ACC_W]) <= 0)
                            c3_act[gf][r][c] = '0;
                        else if ($signed(raw_flat[(r*10+c)*C3_ACC_W +: C3_ACC_W]) > MAX_POS)
                            c3_act[gf][r][c] = MAX_POS;
                        else
                            c3_act[gf][r][c] = raw_flat[(r*10+c)*C3_ACC_W +: DATA_WIDTH];
                    end
        end
    endgenerate

    // =====================================================================
    //  S4: 16 x avg_pool(10x10 -> 5x5)
    // =====================================================================
    logic signed [DATA_WIDTH-1:0] s4_out [0:15][0:4][0:4];

    generate
        for (gf = 0; gf < 16; gf++) begin : gen_s4
            logic signed [DATA_WIDTH-1:0] pool_in [0:9][0:9];
            logic [DATA_WIDTH*5*5-1:0] pool_flat;
            logic rdy;

            always_comb
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 10; c++)
                        pool_in[r][c] = c3_act[gf][r][c];

            pool #(
                .DATA_WIDTH(DATA_WIDTH), .IN_H(10), .IN_W(10),
                .POOL_SIZE(2), .MODE(0)
            ) u_pool (
                .clk(clk), .rst_n(rst_n), .start(s4_go),
                .i(pool_in), .result_flat(pool_flat), .valid(rdy)
            );

            assign s4_done_vec[gf] = rdy;

            always_comb
                for (int r = 0; r < 5; r++)
                    for (int c = 0; c < 5; c++)
                        s4_out[gf][r][c] = $signed(pool_flat[(r*5+c)*DATA_WIDTH +: DATA_WIDTH]);
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
                    fc5_in[ch*25 + r*5 + c] = s4_out[ch][r][c];

    // =====================================================================
    //  FC5: dense(400 -> 120) + saturating ReLU
    // =====================================================================
    logic [FC5_ACC_W*120-1:0] fc5_flat;
    logic signed [DATA_WIDTH-1:0] fc6_in [0:119];

    dense #(
        .DATA_WIDTH(DATA_WIDTH), .IN_FEATURES(400), .OUT_FEATURES(120)
    ) u_fc5 (
        .clk(clk), .rst_n(rst_n), .start(fc5_go),
        .in_data(fc5_in), .weight(fc5_weight), .bias(fc5_bias),
        .out_flat(fc5_flat), .valid(fc5_done)
    );

    always_comb
        for (int j = 0; j < 120; j++)
            if ($signed(fc5_flat[j*FC5_ACC_W +: FC5_ACC_W]) <= 0)
                fc6_in[j] = '0;
            else if ($signed(fc5_flat[j*FC5_ACC_W +: FC5_ACC_W]) > MAX_POS)
                fc6_in[j] = MAX_POS;
            else
                fc6_in[j] = fc5_flat[j*FC5_ACC_W +: DATA_WIDTH];

    // =====================================================================
    //  FC6: dense(120 -> 84) + saturating ReLU
    // =====================================================================
    logic [FC6_ACC_W*84-1:0] fc6_flat;
    logic signed [DATA_WIDTH-1:0] fc7_in [0:83];

    dense #(
        .DATA_WIDTH(DATA_WIDTH), .IN_FEATURES(120), .OUT_FEATURES(84)
    ) u_fc6 (
        .clk(clk), .rst_n(rst_n), .start(fc6_go),
        .in_data(fc6_in), .weight(fc6_weight), .bias(fc6_bias),
        .out_flat(fc6_flat), .valid(fc6_done)
    );

    always_comb
        for (int j = 0; j < 84; j++)
            if ($signed(fc6_flat[j*FC6_ACC_W +: FC6_ACC_W]) <= 0)
                fc7_in[j] = '0;
            else if ($signed(fc6_flat[j*FC6_ACC_W +: FC6_ACC_W]) > MAX_POS)
                fc7_in[j] = MAX_POS;
            else
                fc7_in[j] = fc6_flat[j*FC6_ACC_W +: DATA_WIDTH];

    // =====================================================================
    //  FC7: dense(84 -> 10) — no ReLU on output
    // =====================================================================
    dense #(
        .DATA_WIDTH(DATA_WIDTH), .IN_FEATURES(84), .OUT_FEATURES(10)
    ) u_fc7 (
        .clk(clk), .rst_n(rst_n), .start(fc7_go),
        .in_data(fc7_in), .weight(fc7_weight), .bias(fc7_bias),
        .out_flat(scores_flat), .valid(fc7_done)
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
                    done <= 1'b0; layer_started <= 1'b0;
                    if (start) state <= L_C1;
                end
                L_C1: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (c1_all_done) begin state <= L_S2; layer_started <= 1'b0; end
                end
                L_S2: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (s2_all_done) begin state <= L_C3; layer_started <= 1'b0; end
                end
                L_C3: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (c3_all_done) begin state <= L_S4; layer_started <= 1'b0; end
                end
                L_S4: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (s4_all_done) begin state <= L_FC5; layer_started <= 1'b0; end
                end
                L_FC5: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc5_done) begin state <= L_FC6; layer_started <= 1'b0; end
                end
                L_FC6: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc6_done) begin state <= L_FC7; layer_started <= 1'b0; end
                end
                L_FC7: begin
                    if (!layer_started) layer_started <= 1'b1;
                    else if (fc7_done) begin state <= L_DONE; layer_started <= 1'b0; end
                end
                L_DONE: begin
                    done <= 1'b1; state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
