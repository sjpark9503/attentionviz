import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def head_view(attention, token_a, token_b=None, prettify_tokens=True):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            token_a: list of token_a
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    if token_b is not None:
        vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="self">self</option>
              <option value="cross">cross</option>
            </select>
            </span>
        <div id='vis'></div>
        """
    else:
        vis_html = """
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div> 
            """
        
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

    if prettify_tokens:
        token_a = format_special_chars(token_a)
        if token_b is not None:
            token_b = format_special_chars(token_b)

    attn = format_attention(attention)

    attn_data = {'self':
        {
            'attn': attn['self'].tolist(),
            'left_text': token_a,
            'right_text': token_a
        }
    }
    if token_b is not None:
        attn_data['cross'] = {
            'attn': attn['cross'].tolist(),
            'left_text': token_a,
            'right_text': token_b,
        }
    params = {
        'attention': attn_data,
        'default_filter': "self"
    }

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))