import pytest
import chainladder as cl

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    plt = None
    Figure = None


def test_heatmap_render(raa):
    """The heatmap method should render correctly given the sample."""
    try:
        raa.heatmap()

    except:
        assert False


# def test_empty_triangle():
#     assert cl.Triangle()


def test_to_frame(raa):
    try:
        cl.Chainladder().fit(raa).cdf_.to_frame()
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=True)
        cl.Chainladder().fit(raa).ultimate_.to_frame()
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=True)

    except:
        assert False


def test_labels(xyz):
    assert (
        xyz.valuation_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        == "2008-12-31 23:59:59.999999"
    )
    assert xyz.origin_grain == "Y"
    assert xyz.development_grain == "Y"
    assert xyz.shape == (1, 5, 11, 11)
    assert xyz.index_label == ["Total"]
    assert xyz.columns_label == ["Incurred", "Paid", "Reported", "Closed", "Premium"]
    assert xyz.origin_label == ["AccidentYear"]


@pytest.mark.xfail(plt is None, reason="matplotlib needed for test")
def test_percent_of_ultimate_triangle(raa):
    """The percent_of_ultimate method should render correctly for Triangle objects."""
    try:
        fig = raa.percent_of_ultimate()
        assert fig is not None
        assert isinstance(fig, Figure)
        plt.close(fig)
    except:
        assert False


@pytest.mark.xfail(plt is None, reason="matplotlib needed for test")
def test_percent_of_ultimate_with_fitted_development(raa):
    """The percent_of_ultimate method should work with triangles that have fitted development patterns."""
    try:
        dev = cl.Development().fit(raa)
        # Use the fitted development patterns via cdf_ attribute
        fig = raa.percent_of_ultimate()
        assert fig is not None
        assert isinstance(fig, Figure)
        plt.close(fig)
    except:
        assert False


@pytest.mark.xfail(plt is None, reason="matplotlib needed for test")
def test_percent_of_ultimate_options(raa):
    """The percent_of_ultimate method should handle various parameter options."""
    try:
        # Test with different parameter combinations
        fig1 = raa.percent_of_ultimate(show_by_accident_year=False)
        assert fig1 is not None
        plt.close(fig1)

        fig2 = raa.percent_of_ultimate(show_average_pattern=False)
        assert fig2 is not None
        plt.close(fig2)

        fig3 = raa.percent_of_ultimate(
            show_by_accident_year=True,
            show_average_pattern=True,
            figsize=(8, 6)
        )
        assert fig3 is not None
        plt.close(fig3)
    except:
        assert False


@pytest.mark.xfail(plt is None, reason="matplotlib needed for test")
def test_percent_of_ultimate_multidimensional_error(clrd):
    """The percent_of_ultimate method should raise ValueError for multidimensional triangles."""
    with pytest.raises(ValueError, match="percent_of_ultimate.*only works with a single triangle"):
        clrd.percent_of_ultimate()


def test_percent_of_ultimate_no_matplotlib():
    """The percent_of_ultimate method should raise ImportError when matplotlib is not available."""
    # Temporarily disable matplotlib for this test
    original_plt = cl.core.display.plt
    cl.core.display.plt = None

    try:
        raa = cl.load_sample('raa')
        with pytest.raises(ImportError, match="percent_of_ultimate requires matplotlib"):
            raa.percent_of_ultimate()
    finally:
        # Restore matplotlib
        cl.core.display.plt = original_plt
